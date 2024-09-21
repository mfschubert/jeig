"""Tests for jax-wrapped eigendecomposition."""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

import jeig
from jeig import jeig as _jeig

jax.config.update("jax_enable_x64", True)


BACKENDS = [
    jeig.EigBackend.JAX,
    jeig.EigBackend.NUMPY,
    jeig.EigBackend.SCIPY,
    jeig.EigBackend.TORCH,
    jeig.EigBackend.TORCH_64,
]

SHAPES = [(1, 16, 16), (2, 16, 16), (2, 64, 64)]


class Backend64ComparisonTest(unittest.TestCase):
    @parameterized.expand(itertools.product(BACKENDS, SHAPES))
    def test_backends_against_jax_symmetric_real(self, backend, shape):
        matrix = jax.random.normal(jax.random.PRNGKey(0), shape)
        matrix = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))

        expected_eigval, expected_eigvec = jeig.eig(matrix, backend=jeig.EigBackend.JAX)
        eigval, eigvec = jeig.eig(matrix, backend=backend)

        onp.testing.assert_allclose(eigval, expected_eigval)
        onp.testing.assert_allclose(eigvec, expected_eigvec)

    @parameterized.expand(itertools.product(BACKENDS, SHAPES))
    def test_backends_against_jax_real(self, backend, shape):
        matrix = jax.random.normal(jax.random.PRNGKey(0), shape)

        expected_eigval, expected_eigvec = jeig.eig(matrix, backend=jeig.EigBackend.JAX)
        eigval, eigvec = jeig.eig(matrix, backend=backend)
        onp.testing.assert_allclose(eigval, expected_eigval, err_msg=backend)
        onp.testing.assert_allclose(eigvec, expected_eigvec, err_msg=backend)

    @parameterized.expand(itertools.product(BACKENDS, SHAPES))
    def test_backends_against_jax_complex(self, backend, shape):
        real, imag = jax.random.normal(jax.random.PRNGKey(0), (2,) + shape)
        matrix = real + 1j * imag

        expected_eigval, expected_eigvec = jeig.eig(matrix, backend=jeig.EigBackend.JAX)
        eigval, eigvec = jeig.eig(matrix, backend=backend)

        onp.testing.assert_allclose(eigval, expected_eigval)
        onp.testing.assert_allclose(eigvec, expected_eigvec)

class Backend32ComparisonTest(unittest.TestCase):
    @parameterized.expand(itertools.product(SHAPES))
    def test_torch32_against_scipy_symmetric_real(self, shape):
        matrix = jax.random.normal(jax.random.PRNGKey(0), shape)
        matrix = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
        matrix = matrix.astype(jnp.float32)

        expected_eigval, expected_eigvec = jeig.eig(matrix, backend=jeig.EigBackend.SCIPY)
        eigval, eigvec = jeig.eig(matrix, backend=jeig.EigBackend.TORCH_32)
        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(eigval, expected_eigval)
        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(eigvec, expected_eigvec)

    @parameterized.expand(itertools.product(SHAPES))
    def test_torch32_against_scipy_real(self, shape):
        matrix = jax.random.normal(jax.random.PRNGKey(0), shape)
        matrix = matrix.astype(jnp.float32)

        expected_eigval, expected_eigvec = jeig.eig(matrix, backend=jeig.EigBackend.SCIPY)
        eigval, eigvec = jeig.eig(matrix, backend=jeig.EigBackend.TORCH_32)
        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(eigval, expected_eigval)
        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(eigvec, expected_eigvec)

    @parameterized.expand(itertools.product(SHAPES))
    def test_torch32_against_scipy_complex(self, shape):
        real, imag = jax.random.normal(jax.random.PRNGKey(0), (2,) + shape)
        matrix = real + 1j * imag
        matrix = matrix.astype(jnp.complex64)

        expected_eigval, expected_eigvec = jeig.eig(matrix, backend=jeig.EigBackend.SCIPY)
        eigval, eigvec = jeig.eig(matrix, backend=jeig.EigBackend.TORCH_32)

        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(eigval, expected_eigval)
        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(eigvec, expected_eigvec) 

class BackendSetterTest(unittest.TestCase):
    @parameterized.expand(BACKENDS)
    def test_set_backend(self, backend):
        self.assertEqual(_jeig._DEFAULT_BACKEND, jeig.EigBackend.TORCH)
        jeig.set_backend(backend)
        self.assertEqual(_jeig._DEFAULT_BACKEND, backend)
        jeig.set_backend(jeig.EigBackend.TORCH)
        self.assertEqual(_jeig._DEFAULT_BACKEND, jeig.EigBackend.TORCH)

    @parameterized.expand(["jax", "numpy", "scipy", "torch"])
    def test_set_backend_with_str(self, backend_str):
        self.assertEqual(_jeig._DEFAULT_BACKEND, jeig.EigBackend.TORCH)
        jeig.set_backend(backend_str)
        self.assertEqual(
            _jeig._DEFAULT_BACKEND, jeig.EigBackend.from_string(backend_str)
        )
        jeig.set_backend(jeig.EigBackend.TORCH)
        self.assertEqual(_jeig._DEFAULT_BACKEND, jeig.EigBackend.TORCH)
