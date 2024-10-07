import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)

        return jax.nn.softmax(x)

class MLPPolicy(nn.Module):
    num_actions: int
    num_agents: int

    @nn.compact
    def __call__(self, x):
        mlp = nn.vmap(
            MLP,
            in_axes=0, out_axes=0,
            axis_size=self.num_agents,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )(self.num_actions)

        return mlp(x)