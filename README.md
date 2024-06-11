# jeig - Eigendecompositions wrapped for jax
`v0.0.2`

## Overview

This package wraps eigendecompositions as provided by jax, numpy, scipy, and torch for use with jax. Depending upon your system and your versions of these packages, you may observe significant speed differences.

The wrapped `eig` function also includes a custom vjp rule so that gradients with respect to eigenvalues and eigenvectors can be computed.

![Speed comparison](/docs/speed.png)

## Install
jeig can be installed via pip,
```
pip install jeig
```
This will also install torch. If you only need torch for use with jeig, then the CPU-only version is sufficient and you may wish to install manually as described in the [pytorch docs](https://pytorch.org/get-started/locally/).

## Example usage

```python
import jax
import jeig.eig as jeig

matrix = jax.random.normal(jax.random.PRNGKey(0), (8, 320, 320))

jeig.BACKEND_EIG = jeig.JAX
%timeit jeig.eig(matrix)

jeig.BACKEND_EIG = jeig.NUMPY
%timeit jeig.eig(matrix)

jeig.BACKEND_EIG = jeig.SCIPY
%timeit jeig.eig(matrix)

jeig.BACKEND_EIG = jeig.TORCH
%timeit jeig.eig(matrix)
```
```
376 ms ± 11.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
689 ms ± 11.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
414 ms ± 19 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
136 ms ± 4 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Credit
The high-level `eig` function and the tests are adapted from [fmmax](https://github.com/facebookresearch/fmmax/tree/main/src/fmmax). The torch implementation of eigendecomposition is due to a [comment](https://github.com/google/jax/issues/10180#issuecomment-1092098074) by @YouJiacheng.
