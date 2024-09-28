import jax.numpy as jnp
import flax.linen as nn

import math

class DirectParameterization(nn.Module):
    obs_dims:int
    num_actions:int

    @nn.compact
    def __call__(self, x):
        params = self.param("params", lambda _, shape: jnp.full(shape, 1 / self.num_actions), (math.prod(self.obs_dims), self.num_actions))
        idx = jnp.ravel_multi_index(x, self.obs_dims[1:], mode="clip")

        return params[idx]  

class DirectPolicy(nn.Module):

    obs_dims: int
    num_actions: int
    num_agents: int

    @nn.compact
    def __call__(self, x):

        params = nn.vmap(
            DirectParameterization,
            in_axes=0, out_axes=0,
            axis_size=self.num_agents,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )(self.obs_dims, self.num_actions)

        return params(x)