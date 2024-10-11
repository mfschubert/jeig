# jeig - Eigendecompositions wrapped for jax
`v0.1.3`

## Overview

This package wraps eigendecompositions as provided by jax, numpy, scipy, and torch for use with jax. Depending upon your system and your versions of these packages, you may observe significant speed differences.

![Speed comparison](https://github.com/mfschubert/jeig/blob/main/docs/speed.png?raw=true)

## Install
jeig can be installed via pip,
```
pip install jeig
```
This will also install torch. If you only need torch for use with jeig, then the CPU-only version could be sufficient and you may wish to install manually as described in the [pytorch docs](https://pytorch.org/get-started/locally/).

## Example usage

```python
import jax
import jeig

matrix = jax.random.normal(jax.random.PRNGKey(0), (8, 320, 320))

jeig.set_backend("jax")
%timeit jeig.eig(matrix)

jeig.set_backend("numpy")
%timeit jeig.eig(matrix)

jeig.set_backend("scipy")
%timeit jeig.eig(matrix)

jeig.set_backend("torch")
%timeit jeig.eig(matrix)
```
```
916 ms ± 101 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
1.47 s ± 165 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
782 ms ± 75.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
150 ms ± 10.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## Credit
The torch implementation of eigendecomposition is due to a [comment](https://github.com/google/jax/issues/10180#issuecomment-1092098074) by @YouJiacheng.
