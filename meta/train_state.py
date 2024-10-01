import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState


@dataclass
class TrainState:
    team_train_state: TrainState
    adv_train_state: TrainState

    def get_actions(self, rng, obs):

        team_action_probs = self.team_train_state.apply_fn(self.team_train_state.params, obs[:-1])
        adv_action_probs = self.team_train_state.apply_fn(self.adv_train_state.params, obs[-1][None, ...])[0]

        # --- Sample adversary action ---
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(rng, team_action_probs.shape[0])
        team_actions = jax.vmap(jax.random.choice, in_axes=(0, None, None, None, 0))(_rng, team_action_probs.shape[-1], (), True, team_action_probs)

        # --- Sample adversary action ---
        rng, _rng = jax.random.split(rng)
        adv_action = jax.random.choice(_rng, adv_action_probs.shape[-1], (1, ), p=adv_action_probs)

        return jnp.concatenate((team_actions, adv_action))