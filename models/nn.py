import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from functools import partial

def one_hot_fn(obs, obs_dims):

    if len(obs_dims) == 1:
        return jax.nn.one_hot(obs, obs_dims[0])
    else:
        obs = jnp.concatenate(
            [
                jax.nn.one_hot(obs[idx], obs_dims[idx]) for idx, obs_ in enumerate(obs_dims)
            ]
        )
        return obs

class MLP(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)

        return jax.nn.softmax(x).squeeze()

class MLPPolicy(nn.Module):
    obs_dims: tuple[int]
    num_actions: int
    num_agents: int

    @nn.compact
    def __call__(self, x):

        encode_fn = partial(one_hot_fn, obs_dims=self.obs_dims[1:])
        obs = jax.vmap(encode_fn, in_axes=(0))(x)

        mlp = nn.vmap(
            MLP,
            in_axes=0, out_axes=0,
            axis_size=self.num_agents,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )(self.num_actions)

        out = mlp(obs)

        return out