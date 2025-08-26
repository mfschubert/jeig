"""Various implementations of `eig` wrapped for use with jax."""

import enum
import functools
import multiprocessing as mp
import os
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

if torch.cuda.has_magma:
    # Use magma that ships with torch.
    os.environ["JAX_GPU_MAGMA_PATH"] = os.path.join(
        os.path.dirname(torch.__file__), "lib", "libtorch_cuda_linalg.so"
    )

_JAX_HAS_MAGMA = torch.cuda.has_magma

callback = functools.partial(jax.pure_callback, vmap_method="expand_dims")
callback_sequential = functools.partial(jax.pure_callback, vmap_method="sequential")


NDArray = onp.ndarray[Any, Any]


@enum.unique
class EigBackend(enum.Enum):
    JAX = "jax"
    MAGMA = "magma"
    NUMPY = "numpy"
    SCIPY = "scipy"
    TORCH = "torch"

    @classmethod
    def from_string(cls, backend: str) -> "EigBackend":
        """Returns the specified backend."""
        return {
            cls.JAX.value: cls.JAX,
            cls.MAGMA.value: cls.MAGMA,
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
    matrix: jnp.ndarray,
    backend: Optional[str | EigBackend] = None,
    force_x64: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using the specified backend.

    If no backend is `None`, the backend specified by the module-level constant
    `_DEFAULT_BACKEND` is used. This can be updated with the `set_backend` function.

    Args:
        matrix: The matrix for which the eigendecomposition is sought. May have
            arbitrary batch dimensions.
        backend: Optional string to specify the backend.
        force_x64: If `True` and `matrix` has 32-bit precision, it will be cast to
            64-bit for the eigendecomposition. The eigenvectors and eigenvalues will
            be cast to 32-bit before returning. Requires 64-bit jax to be enabled.

    Returns:
        The eigenvalues and right eigenvectors of the matrix.
    """
    with jax.ensure_compile_time_eval():
        if backend is None:
            backend = _DEFAULT_BACKEND
        elif not isinstance(backend, EigBackend):
            backend = EigBackend.from_string(backend)
    return EIG_FNS[backend](matrix, force_x64)


def _eig_jax(matrix: jnp.ndarray, force_x64: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.numpy.linalg.eig`."""
    if force_x64:
        matrix_dtype = jnp.promote_types(matrix.dtype, jnp.float64)
    else:
        matrix_dtype = matrix.dtype
    output_dtype = jnp.promote_types(matrix, jnp.complex64)
    eigval, eigvec = jax.lax.linalg.eig(
        matrix.astype(matrix_dtype),
        compute_left_eigenvectors=False,
        use_magma=False,
    )
    return eigval.astype(output_dtype), eigvec.astype(output_dtype)


def _eig_magma(matrix: jnp.ndarray, force_x64: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.numpy.linalg.eig`."""
    if not _JAX_HAS_MAGMA:
        raise ValueError(
            "`MAGMA` backend is not available; `torch.cuda.has_magma` is `False`."
        )
    if force_x64:
        matrix_dtype = jnp.promote_types(matrix.dtype, jnp.float64)
    else:
        matrix_dtype = matrix.dtype
    output_dtype = jnp.promote_types(matrix, jnp.complex64)
    eigval, eigvec = jax.lax.linalg.eig(
        matrix.astype(matrix_dtype),
        compute_left_eigenvectors=False,
        use_magma=True,
    )
    return eigval.astype(output_dtype), eigvec.astype(output_dtype)


def _eig_numpy(matrix: jnp.ndarray, force_x64: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `numpy.linalg.eig`."""

    def _numpy_eig_fn(matrix: NDArray) -> Tuple[NDArray, NDArray]:
        if force_x64:
            matrix_dtype = onp.promote_types(matrix.dtype, onp.float64)
        else:
            matrix_dtype = matrix.dtype
        output_dtype = onp.promote_types(matrix, onp.complex64)
        eigval, eigvec = onp.linalg.eig(matrix.astype(matrix_dtype))
        return eigval.astype(output_dtype), eigvec.astype(output_dtype)

    dtype = jnp.promote_types(matrix.dtype, jnp.complex64)
    eigval, eigvec = callback(
        _numpy_eig_fn,
        (
            jnp.ones(matrix.shape[:-1], dtype=dtype),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=dtype),  # Eigenvectors
        ),
        matrix.astype(dtype),
    )
    return jnp.asarray(eigval), jnp.asarray(eigvec)


def _eig_scipy(matrix: jnp.ndarray, force_x64: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `scipy.linalg.eig`."""

    def _scipy_eig_fn(matrix: NDArray) -> Tuple[NDArray, NDArray]:
        if force_x64:
            matrix_dtype = onp.promote_types(matrix.dtype, onp.float64)
        else:
            matrix_dtype = matrix.dtype
        output_dtype = onp.promote_types(matrix, onp.complex64)
        eigval, eigvec = scipy.linalg.eig(matrix.astype(matrix_dtype))
        return eigval.astype(output_dtype), eigvec.astype(output_dtype)

    def _eig_fn(m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dtype = jnp.promote_types(matrix.dtype, jnp.complex64)
        eigval, eigvec = callback_sequential(
            _scipy_eig_fn,
            (
                jnp.ones(m.shape[:-1], dtype=dtype),  # Eigenvalues
                jnp.ones(m.shape, dtype=dtype),  # Eigenvectors
            ),
            m.astype(dtype),
        )
        return jnp.asarray(eigval), jnp.asarray(eigvec)

    batch_shape = matrix.shape[:-2]
    matrix = jnp.reshape(matrix, (-1,) + matrix.shape[-2:])
    eigval, eigvec = jax.vmap(_eig_fn)(matrix)
    eigvec = jnp.reshape(eigvec, batch_shape + eigvec.shape[-2:])
    eigval = jnp.reshape(eigval, batch_shape + eigval.shape[-1:])
    return eigval, eigvec


def _eig_torch(matrix: jnp.ndarray, force_x64: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `torc.linalg.eig`."""
    dtype = jnp.promote_types(matrix.dtype, jnp.complex64)

    def _eig_fn(matrix: jnp.ndarray) -> Tuple[NDArray, NDArray]:
        if force_x64:
            matrix_dtype = onp.promote_types(matrix.dtype, onp.float64)
        else:
            matrix_dtype = matrix.dtype
        output_dtype = onp.promote_types(matrix, onp.complex64)
        batch_shape = matrix.shape[:-2]
        matrix_flat = onp.array(matrix, dtype=matrix_dtype).reshape(
            (-1,) + matrix.shape[-2:]
        )
        results = _eig_torch_parallelized(torch.as_tensor(matrix_flat))
        eigval = onp.asarray([eigval.numpy() for eigval, _ in results], dtype=dtype)
        eigvec = onp.asarray([eigvec.numpy() for _, eigvec in results], dtype=dtype)
        eigval = eigval.reshape(batch_shape + (eigval.shape[-1],))
        eigvec = eigvec.reshape(batch_shape + eigvec.shape[-2:])
        return eigval.astype(output_dtype), eigvec.astype(output_dtype)

    eigval, eigvec = callback(
        _eig_fn,
        (
            jnp.ones(matrix.shape[:-1], dtype=dtype),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=dtype),  # Eigenvectors
        ),
        matrix.astype(dtype),
    )
    return eigval, eigvec


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
    EigBackend.MAGMA: _eig_magma,
    EigBackend.NUMPY: _eig_numpy,
    EigBackend.SCIPY: _eig_scipy,
    EigBackend.TORCH: _eig_torch,
}
