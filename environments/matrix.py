import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment
from gymnax.environments import spaces
from flax.struct import dataclass

@dataclass
class EnvState(environment.EnvState):
    current_state: int
    time: int

@dataclass
class EnvParams(environment.EnvParams):
    all_utilities: jnp.ndarray = None
    transitions: jnp.ndarray = None

ENV_CONFIG = {
    "num_states": 3,
    "num_agents": 3,
    "num_actions": 3,
    "max_time": 8
}

class AdvMatrix(environment.Environment):

    def __init__(self, **kwargs):
        super().__init__()

        self.num_states = kwargs["num_states"]
        self.num_agents = kwargs["num_agents"]
        self._num_actions = kwargs["num_actions"]

        self.max_time = kwargs["max_time"]

    @property
    def num_actions(self):
        return self._num_actions

    def action_space(self, params: EnvParams = None):
        return spaces.Tuple(
            tuple(spaces.Discrete(self.num_actions) for _ in range(self.num_agents))
        )
    
    def observation_space(self, params: EnvParams = None):
        return spaces.Tuple((spaces.Discrete(self.num_agents), spaces.Discrete(1)))

    @property
    def default_params(self) -> EnvParams:
        rng = jax.random.key(0)

        rng, _rng = jax.random.split(rng)
        action_shape = np.pow(self.num_actions, self.num_agents)
        all_utilities = jax.random.uniform(_rng, (self.num_states, action_shape), minval=-1, maxval=1)

        rng, _rng = jax.random.split(rng)
        transitions = jax.random.uniform(_rng, (action_shape, self.num_states))
        transitions /= transitions.sum(axis=1)[..., None]

        return EnvParams(
            max_steps_in_episode=self.max_time,
            all_utilities=all_utilities,
            transitions=transitions
        )
    
    def reset_env(
        self, 
        rng,
        params: EnvParams
    ):

        rng, _rng = jax.random.split(rng)
        obs = jax.random.choice(_rng, self.num_states)
        full_obs = jnp.full((self.num_agents, 1), obs, dtype=int)

        state = EnvState(
            obs, 0
        )

        return full_obs, state
    
    def step_env(
        self,
        rng,
        state: EnvState,
        action: int,
        params: EnvParams,
    ):
        # --- Calculate utility ---
        action_idx = jnp.ravel_multi_index(action, tuple(self.num_actions for _ in range(self.num_agents)), mode="clip")
        utility = params.all_utilities[state.current_state, action_idx]
        full_utility = jnp.full((self.num_agents,), utility).at[-1].set(-utility) * (state.time <= self.max_time)

        # --- Calculate new state ---
        rng, _rng = jax.random.split(rng)
        obs = jax.random.choice(_rng, self.num_states, p = params.transitions[action_idx])
        full_obs = jnp.full((self.num_agents, 1), obs, dtype=int)

        state = EnvState(obs, state.time + 1)

        done = state.time > self.max_time

        return (
            jax.lax.stop_gradient(full_obs),
            jax.lax.stop_gradient(state),
            full_utility,
            done,
            {},
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "AdvMatrix-v1"



