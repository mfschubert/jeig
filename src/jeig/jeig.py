"""Various implementations of `eig` wrapped for use with jax."""

import enum
import multiprocessing as mp
import warnings
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import scipy  # type: ignore[import-untyped]
import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(mp.cpu_count())
except RuntimeError as exc:
    warnings.warn(str(exc))

NDArray = onp.ndarray[Any, Any]


@enum.unique
class EigBackend(enum.Enum):
    JAX = "jax"
    NUMPY = "numpy"
    SCIPY = "scipy"
    TORCH = "torch"

    @classmethod
    def from_string(cls, backend: str) -> "EigBackend":
        """Returns the specified backend."""
        return {
            cls.JAX.value: cls.JAX,
            cls.NUMPY.value: cls.NUMPY,
            cls.SCIPY.value: cls.SCIPY,
            cls.TORCH.value: cls.TORCH,
        }[backend]


_DEFAULT_BACKEND = EigBackend.TORCH


def set_backend(backend: str | EigBackend) -> None:
    """Sets the backend for eigendecomposition."""
    if not isinstance(backend, EigBackend):
        backend = EigBackend.from_string(backend)
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend


def eig(
    matrix: jnp.ndarray, backend: Optional[str] | EigBackend = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using the specified backend.

    If no backend is `None`, the backend specified by the module-level constant
    `_DEFAULT_BACKEND` is used. This can be updated with the `set_backend`.

    Args:
        matrix: The matrix for which the eigendecomposition is sought. May have
            arbitrary batch dimensions.
        backend: Optional string to specify the backend.

    Returns:
        The eigenvalues and eigenvectors of the matrix.
    """
    with jax.ensure_compile_time_eval():
        if backend is None:
            backend = _DEFAULT_BACKEND
        elif not isinstance(backend, EigBackend):
            backend = EigBackend.from_string(backend)
    return EIG_FNS[backend](matrix)


def _eig_jax(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.numpy.linalg.eig`."""
    if jax.devices()[0] == jax.devices("cpu")[0]:
        return jnp.linalg.eig(matrix)
    else:
        eigvals, eigvecs = jax.pure_callback(
            _eig_jax_cpu,
            (
                jnp.ones(matrix.shape[:-1], dtype=complex),  # Eigenvalues
                jnp.ones(matrix.shape, dtype=complex),  # Eigenvectors
            ),
            matrix.astype(complex),
            vectorized=True,
        )
        return jnp.asarray(eigvals), jnp.asarray(eigvecs)


# Define jax eigendecomposition that runs on CPU. Note that the compilation takes
# place at module import time. If the `jit` is inside a function, deadlocks can occur.
with jax.default_device(jax.devices("cpu")[0]):
    _eig_jax_cpu = jax.jit(jnp.linalg.eig)


def _eig_numpy(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `numpy.linalg.eig`."""
    eigval, eigvec = jax.pure_callback(
        onp.linalg.eig,
        (
            jnp.ones(matrix.shape[:-1], dtype=complex),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=complex),  # Eigenvectors
        ),
        matrix.astype(complex),
        vectorized=True,
    )
    return jnp.asarray(eigval), jnp.asarray(eigvec)


def _eig_scipy(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `scipy.linalg.eig`."""

    def _eig_fn(m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        eigval, eigvec = jax.pure_callback(
            scipy.linalg.eig,
            (
                jnp.ones(m.shape[:-1], dtype=complex),  # Eigenvalues
                jnp.ones(m.shape, dtype=complex),  # Eigenvectors
            ),
            m.astype(complex),
            vectorized=False,
        )
        return jnp.asarray(eigval), jnp.asarray(eigvec)

    batch_shape = matrix.shape[:-2]
    matrix = jnp.reshape(matrix, (-1,) + matrix.shape[-2:])
    eigvals, eigvecs = jax.vmap(_eig_fn)(matrix)
    eigvecs = jnp.reshape(eigvecs, batch_shape + eigvecs.shape[-2:])
    eigvals = jnp.reshape(eigvals, batch_shape + eigvals.shape[-1:])
    return eigvals, eigvecs


def _eig_torch(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `torc.linalg.eig`."""

    def _eig_fn(matrix: jnp.ndarray) -> Tuple[NDArray, NDArray]:
        results = _eig_torch_parallelized(torch.as_tensor(onp.array(matrix)))
        eigvals = onp.asarray([eigval.numpy() for eigval, _ in results])
        eigvecs = onp.asarray([eigvec.numpy() for _, eigvec in results])
        return eigvals.astype(matrix.dtype), eigvecs.astype(matrix.dtype)

    batch_shape = matrix.shape[:-2]
    matrix = jnp.reshape(matrix, (-1,) + matrix.shape[-2:])
    assert matrix.ndim == 3
    eigvals, eigvecs = jax.pure_callback(
        _eig_fn,
        (
            jnp.ones(matrix.shape[:-1], dtype=complex),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=complex),  # Eigenvectors
        ),
        matrix.astype(complex),
        vectorized=True,
    )
    eigvecs = jnp.reshape(eigvecs, batch_shape + eigvecs.shape[-2:])
    eigvals = jnp.reshape(eigvals, batch_shape + eigvals.shape[-1:])
    return eigvals, eigvecs


@torch.jit.script
def _eig_torch_parallelized(x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Parallelized eigendecomposition using torch."""
    # See https://github.com/google/jax/issues/10180#issuecomment-1092098074
    futures = [
        torch.jit.fork(torch.linalg.eig, x[i])  # type: ignore[no-untyped-call]
        for i in range(x.shape[0])
    ]
    return [torch.jit.wait(fut) for fut in futures]  # type: ignore[no-untyped-call]


EIG_FNS = {
    EigBackend.JAX: _eig_jax,
    EigBackend.NUMPY: _eig_numpy,
    EigBackend.SCIPY: _eig_scipy,
    EigBackend.TORCH: _eig_torch,
}
