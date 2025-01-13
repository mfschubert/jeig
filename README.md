# jeig - Eigendecompositions wrapped for jax
`v0.2.3`

## Overview

This package wraps eigendecompositions as provided by jax, magma, numpy, scipy, and torch for use with jax. Depending upon your system and your versions of these packages, you may observe significant speed differences. The following were obtained using jax 0.4.37 on a system with 28-core Intel Xeon w7-3465X and NVIDIA RTX4090.

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

matrix = jax.random.normal(jax.random.PRNGKey(0), (16, 1024, 1024))

%timeit jax.block_until_ready(jeig.eig(matrix, backend="jax"))

%timeit jax.block_until_ready(jeig.eig(matrix, backend="magma"))

%timeit jax.block_until_ready(jeig.eig(matrix, backend="numpy"))

%timeit jax.block_until_ready(jeig.eig(matrix, backend="scipy"))

%timeit jax.block_until_ready(jeig.eig(matrix, backend="torch"))
```
```
6.81 s ± 54.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
1min 15s ± 1.35 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
28.6 s ± 341 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
14.8 s ± 396 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
1.43 s ± 77.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## Credit
The torch implementation of eigendecomposition is due to a [comment](https://github.com/google/jax/issues/10180#issuecomment-1092098074) by @YouJiacheng.
