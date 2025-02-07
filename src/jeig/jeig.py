"""Various implementations of `eig` wrapped for use with jax."""

import enum
import functools
import multiprocessing as mp
import os
import warnings
from packaging import version
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import scipy  # type: ignore[import-untyped]
import torch
from jax.experimental import shard_map

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

_JAX_SUPPORTS_GPU_EIG = version.Version(jax.__version__) >= version.Version("0.4.36")
_JAX_HAS_GPU = jax.devices()[0] != jax.devices("cpu")[0]
_TORCH_HAS_MAGMA = torch.cuda.has_magma

if version.Version(jax.__version__) > version.Version("0.4.31"):
    callback = functools.partial(jax.pure_callback, vmap_method="expand_dims")
    callback_sequential = functools.partial(jax.pure_callback, vmap_method="sequential")
else:
    callback = functools.partial(jax.pure_callback, vectorized=True)
    callback_sequential = functools.partial(jax.pure_callback, vectorized=False)


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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using the specified backend.

    If no backend is `None`, the backend specified by the module-level constant
    `_DEFAULT_BACKEND` is used. This can be updated with the `set_backend` function.

    Args:
        matrix: The matrix for which the eigendecomposition is sought. May have
            arbitrary batch dimensions.
        backend: Optional string to specify the backend.

    Returns:
        The eigenvalues and right eigenvectors of the matrix.
    """
    with jax.ensure_compile_time_eval():
        if backend is None:
            backend = _DEFAULT_BACKEND
        elif not isinstance(backend, EigBackend):
            backend = EigBackend.from_string(backend)
    return EIG_FNS[backend](matrix)


def _eig_jax(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.numpy.linalg.eig`."""
    if _JAX_SUPPORTS_GPU_EIG:
        return jax.lax.linalg.eig(
            matrix,
            compute_left_eigenvectors=False,
            use_magma=False,
        )
    elif not _JAX_HAS_GPU:
        return jnp.linalg.eig(matrix)
    else:
        # Older jax versions on GPU require some gymnastics to run on CPU.
        dtype = jnp.promote_types(matrix.dtype, jnp.complex64)
        eigenvalues, eigenvectors = callback(
            _jit_jnp_linalg_eig,
            (
                jnp.ones(matrix.shape[:-1], dtype=dtype),  # Eigenvalues
                jnp.ones(matrix.shape, dtype=dtype),  # Eigenvectors
            ),
            matrix.astype(dtype),
        )
        return eigenvalues, eigenvectors


_jit_jnp_linalg_eig = jax.jit(jnp.linalg.eig)


def _eig_jax_multicore(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.numpy.linalg.eig` with multicore support."""
    cpu_count = len(jax.devices("cpu"))

    # If there is only one CPU, or if on GPU backend with a jax version that does not
    # support directly calling `jax.lax.linalg.eig`, fall back to single core.
    if cpu_count == 1 or (_JAX_HAS_GPU and not _JAX_SUPPORTS_GPU_EIG):
        return _eig_jax(matrix)

    with jax.ensure_compile_time_eval():
        batch_shape = matrix.shape[:-2]
        matrix_shape = matrix.shape[-2:]
        batch_size = int(onp.prod(batch_shape))
        num_parallel = min(batch_size, cpu_count)

        mesh = jax.make_mesh((min(batch_size, cpu_count),), ("parallel",))
        pspec = jax.sharding.PartitionSpec("parallel")
        sharding = jax.sharding.NamedSharding(mesh, pspec)

        if batch_size > num_parallel:
            num_pad = (cpu_count - batch_size % cpu_count) % cpu_count
        else:
            num_pad = 0

    matrix = matrix.reshape((-1,) + matrix_shape)
    matrix = jnp.pad(matrix, ((0, num_pad), (0, 0), (0, 0)))
    matrix = matrix.reshape((num_parallel, -1) + matrix_shape)
    matrix = jax.device_put(matrix, sharding)

    eigval, eigvec = shard_map.shard_map(
        lambda x: jax.lax.linalg.eig(x, compute_left_eigenvectors=False),
        mesh=mesh,
        in_specs=pspec,
        out_specs=[pspec, pspec],
        check_rep=True,
    )(matrix)

    eigval = eigval.reshape((-1,) + eigval.shape[2:])
    eigvec = eigvec.reshape((-1,) + eigvec.shape[2:])
    eigval = eigval[:batch_size]
    eigvec = eigvec[:batch_size]
    eigval = eigval.reshape(batch_shape + eigval.shape[1:])
    eigvec = eigvec.reshape(batch_shape + eigvec.shape[1:])
    return eigval, eigvec


def _eig_magma(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.numpy.linalg.eig`."""
    if not _JAX_SUPPORTS_GPU_EIG:
        raise ValueError(
            f"`MAGMA` backend is not available; jax version {jax.__version__} is less "
            f"than minimum 0.4.36."
        )
    if not _TORCH_HAS_MAGMA:
        raise ValueError(
            "`MAGMA` backend is not available; `torch.cuda.has_magma` is `False`."
        )
    eigval, eigvec = jax.lax.linalg.eig(
        matrix, compute_left_eigenvectors=False, use_magma=True
    )
    return eigval, eigvec


def _eig_numpy(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `numpy.linalg.eig`."""
    dtype = jnp.promote_types(matrix.dtype, jnp.complex64)
    eigval, eigvec = callback(
        onp.linalg.eig,
        (
            jnp.ones(matrix.shape[:-1], dtype=dtype),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=dtype),  # Eigenvectors
        ),
        matrix.astype(dtype),
    )
    return jnp.asarray(eigval), jnp.asarray(eigvec)


def _eig_scipy(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `scipy.linalg.eig`."""

    def _eig_fn(m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dtype = jnp.promote_types(matrix.dtype, jnp.complex64)
        eigval, eigvec = callback_sequential(
            scipy.linalg.eig,
            (
                jnp.ones(m.shape[:-1], dtype=dtype),  # Eigenvalues
                jnp.ones(m.shape, dtype=dtype),  # Eigenvectors
            ),
            m.astype(dtype),
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
    dtype = jnp.promote_types(matrix.dtype, jnp.complex64)

    def _eig_fn(matrix: jnp.ndarray) -> Tuple[NDArray, NDArray]:
        batch_shape = matrix.shape[:-2]
        matrix_flat = onp.array(matrix).reshape((-1,) + matrix.shape[-2:])
        results = _eig_torch_parallelized(torch.as_tensor(matrix_flat))
        eigvals = onp.asarray([eigval.numpy() for eigval, _ in results], dtype=dtype)
        eigvecs = onp.asarray([eigvec.numpy() for _, eigvec in results], dtype=dtype)
        eigvals = eigvals.reshape(batch_shape + (eigvals.shape[-1],))
        eigvecs = eigvecs.reshape(batch_shape + eigvecs.shape[-2:])
        return eigvals, eigvecs

    eigvals, eigvecs = callback(
        _eig_fn,
        (
            jnp.ones(matrix.shape[:-1], dtype=dtype),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=dtype),  # Eigenvectors
        ),
        matrix.astype(dtype),
    )
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
    EigBackend.JAX: _eig_jax_multicore,
    EigBackend.MAGMA: _eig_magma,
    EigBackend.NUMPY: _eig_numpy,
    EigBackend.SCIPY: _eig_scipy,
    EigBackend.TORCH: _eig_torch,
}
