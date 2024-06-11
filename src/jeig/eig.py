"""Various implementations of `eig` wrapped for use with jax."""

import multiprocessing as mp
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import scipy
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(mp.cpu_count())


JAX = "jax"
NUMPY = "numpy"
SCIPY = "scipy"
TORCH = "torch"

BACKEND_EIG = TORCH

EPS_EIG = 1e-6

# -----------------------------------------------------------------------------
# Define the eigendecomposition operation and its custom vjp rule.
# -----------------------------------------------------------------------------


@jax.custom_vjp
def eig(
    matrix: jnp.ndarray,
    eps: float = EPS_EIG,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the eigendecomposition of `matrix`.

    The implementation uses a custom vjp rule to enable calculation of gradients with
    respect to the eigenvalues and eigenvectors. The expression for the gradient is
    from [2019 Boeddeker], and a regularization scheme similar to [20921 Colburn] is
    used.

    Args:
        matrix: The matrix for which eigenvalues and eigenvectors are sought.
        eps: Parameter which determines the degree of broadening.

    Returns:
        The eigenvalues and eigenvectors.
    """
    del eps
    return _eig(matrix)


def _eig_fwd(
    matrix: jnp.ndarray,
    eps: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, float]]:
    """Implements the forward calculation for `eig`."""
    eigenvalues, eigenvectors = _eig(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors, eps)


def _eig_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, float],
    grads: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None]:
    """Implements the backward calculation for `eig`."""
    eigenvalues, eigenvectors, eps = res
    grad_eigenvalues, grad_eigenvectors = grads

    # Compute the broadened F-matrix. The expression is similar to that of equation 5
    # from [2021 Colburn], but differs slightly from both the code and paper.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    delta_eig = eigenvalues_i - eigenvalues_j
    f_broadened = delta_eig.conj() / (jnp.abs(delta_eig) ** 2 + eps)

    # Manually set the diagonal elements to zero, as we do not use broadening here.
    i = jnp.arange(f_broadened.shape[-1])
    f_broadened = f_broadened.at[..., i, i].set(0)

    # By jax convention, gradients are with respect to the complex parameters, not with
    # respect to their conjugates. Take the conjugates.
    grad_eigenvalues_conj = jnp.conj(grad_eigenvalues)
    grad_eigenvectors_conj = jnp.conj(grad_eigenvectors)

    eigenvectors_H = matrix_adjoint(eigenvectors)
    dim = eigenvalues.shape[-1]
    eye_mask = jnp.eye(dim, dtype=bool)
    eye_mask = eye_mask.reshape((1,) * (eigenvalues.ndim - 1) + (dim, dim))

    # Then, the gradient is found by equation 4.77 of [2019 Boeddeker].
    rhs = (
        diag(grad_eigenvalues_conj)
        + jnp.conj(f_broadened) * (eigenvectors_H @ grad_eigenvectors_conj)
        - jnp.conj(f_broadened)
        * (eigenvectors_H @ eigenvectors)
        @ jnp.where(eye_mask, jnp.real(eigenvectors_H @ grad_eigenvectors_conj), 0.0)
    ) @ eigenvectors_H
    grad_matrix = jnp.linalg.solve(eigenvectors_H, rhs)

    # Take the conjugate of the gradient, reverting to the jax convention
    # where gradients are with respect to complex parameters.
    grad_matrix = jnp.conj(grad_matrix)

    # Return `grad_matrix`, and `None` for the gradient with respect to `eps` and
    # `backend`.
    return grad_matrix, None


eig.defvjp(_eig_fwd, _eig_bwd)


def diag(x: jnp.ndarray) -> jnp.ndarray:
    """A batch-compatible version of `numpy.diag`."""
    shape = x.shape + (x.shape[-1],)
    y = jnp.zeros(shape, x.dtype)
    i = jnp.arange(x.shape[-1])
    return y.at[..., i, i].set(x)


def matrix_adjoint(x: jnp.ndarray) -> jnp.ndarray:
    """Computes the adjoint for a batch of matrices."""
    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    return jnp.conj(jnp.transpose(x, axes=axes))


# -----------------------------------------------------------------------------
# Eigendecompositions for all the backends follow.
# -----------------------------------------------------------------------------


def _eig(
    matrix: jnp.ndarray, backend: Optional[str] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using the specified backend.

    If no backend is `None`, the backend specified by the module-level constant
    `BACKEND_EIG` is used.

    Args:
        matrix: The matrix for which the eigendecomposition is sought. May have
            arbitrary batch dimensions.
        backend: Optional string to specify the backend.

    Returns:
        The eigenvalues and eigenvectors of the matrix.
    """
    with jax.ensure_compile_time_eval():
        if backend is None:
            backend = BACKEND_EIG
    return EIG_FNS[backend](matrix)


def _eig_jax(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.numpy.linalg.eig`."""

    def _eig_fn(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        with jax.default_device(jax.devices("cpu")[0]):
            return jax.jit(jnp.linalg.eig)(matrix)

    return jax.pure_callback(
        _eig_fn,
        (
            jnp.ones(matrix.shape[:-1], dtype=complex),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=complex),  # Eigenvectors
        ),
        matrix.astype(complex),
        vectorized=True,
    )


def _eig_numpy(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `numpy.linalg.eig`."""
    return jax.pure_callback(
        onp.linalg.eig,
        (
            jnp.ones(matrix.shape[:-1], dtype=complex),  # Eigenvalues
            jnp.ones(matrix.shape, dtype=complex),  # Eigenvectors
        ),
        matrix.astype(complex),
        vectorized=True,
    )


def _eig_scipy(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `scipy.linalg.eig`."""

    def _eig_fn(m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.pure_callback(
            scipy.linalg.eig,
            (
                jnp.ones(m.shape[:-1], dtype=complex),  # Eigenvalues
                jnp.ones(m.shape, dtype=complex),  # Eigenvectors
            ),
            m.astype(complex),
            vectorized=False,
        )

    batch_shape = matrix.shape[:-2]
    matrix = jnp.reshape(matrix, (-1,) + matrix.shape[-2:])
    eigvals, eigvecs = jax.vmap(_eig_fn)(matrix)
    eigvecs = jnp.reshape(eigvecs, batch_shape + eigvecs.shape[-2:])
    eigvals = jnp.reshape(eigvals, batch_shape + eigvals.shape[-1:])
    return eigvals, eigvecs


def _eig_torch(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `torc.linalg.eig`."""

    def _eig_fn(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        results = _eig_torch_parallelized(torch.as_tensor(onp.asarray(matrix)))
        eigvals = jnp.asarray([eigval.numpy() for eigval, _ in results])
        eigvecs = jnp.asarray([eigvec.numpy() for _, eigvec in results])
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
    futures = [torch.jit._fork(torch.linalg.eig, x[i]) for i in range(x.shape[0])]
    return [torch.jit._wait(fut) for fut in futures]


EIG_FNS = {
    JAX: _eig_jax,
    NUMPY: _eig_numpy,
    SCIPY: _eig_scipy,
    TORCH: _eig_torch,
}
