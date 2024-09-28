import jax
import jax.numpy as jnp
from functools import partial

def projection_simplex_truncated(x, trunc_size) -> jnp.ndarray: 
    """
    Code adapted from 
    https://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html
    To represent truncated simplex projection. Assumes 1D vector. 
    """
    ones = jnp.ones_like(x)
    lambdas = jnp.concatenate((ones * trunc_size - x, ones - x), axis=-1)
    idx = jnp.argsort(lambdas)
    lambdas = jnp.take_along_axis(lambdas, idx, -1)
    active = jnp.cumsum((jnp.float32(idx < x.shape[-1])) * 2 - 1, axis=-1)[..., :-1]
    diffs = jnp.diff(lambdas, n=1, axis=-1)
    left = (ones * trunc_size).sum(axis=-1)
    left = left.reshape(*left.shape, 1)
    totals = left + jnp.cumsum(active*diffs, axis=-1)

    def generate_vmap(counter, func):
        if counter == 0:
            return func
        else:
            return generate_vmap(counter - 1, jax.vmap(func))
                
    i = jnp.expand_dims(generate_vmap(len(totals.shape) - 1, partial(jnp.searchsorted, v=1))(totals), -1)
    lam = (1 - jnp.take_along_axis(totals, i, -1)) / jnp.take_along_axis(active, i, -1) + jnp.take_along_axis(lambdas, i+1, -1)
    return jnp.clip(x + lam, trunc_size, 1)

