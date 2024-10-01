from flax.struct import dataclass

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from math import pow

@dataclass
class EnvState(environment.EnvState):
    agent: jnp.ndarray
    agents_term: jnp.ndarray
    goal: jnp.ndarray
    goals_term: jnp.ndarray
    time: jnp.ndarray

ENV_CONFIG = {
    "DIM": 6,
    "MAX_TIME": 20,
    "NUM_AGENTS": 3,
    "NUM_GOALS": 2
}

@dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 12
    dim: int = 6
    num_agents: int = 3
    num_goals: int = 2

class AdvMultiGrid(environment.Environment):

    def __init__(self, **kwargs):

        self.dim = kwargs["DIM"]
        self.max_time = kwargs["MAX_TIME"]
        self.num_agents = kwargs["NUM_AGENTS"]
        self.num_goals = kwargs["NUM_GOALS"]

        self.num_cells = int(pow(self.dim - 1, 2))

    @property
    def num_actions(self):
        return 5

    def action_space(self, params = None):
        return spaces.Tuple(
            tuple(spaces.Discrete(self.num_actions) for _ in range(self.num_agents))
        )

    def observation_space(self, params = None):
        return spaces.Tuple(
            (spaces.Discrete(self.num_agents), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2))
        )
    
    @property
    def default_params(self):
        return EnvParams(
            dim = self.dim,
            max_steps_in_episode = self.max_time,
            num_agents = self.num_agents,
            num_goals = self.num_goals
        )

    def reset_env(
        self, 
        rng, 
        params: EnvParams = None
    ):
        # Spawn Objects
        rng, _rng = jax.random.split(rng)
        goals = jax.random.choice(_rng, self.num_cells, (self.num_goals, ), replace=False)
        goals_pos = jax.vmap(lambda x, shape: jnp.stack(jnp.unravel_index(x, shape)), in_axes=(0, None))(goals, (self.dim, self.dim))
        
        # Spawn Agents
        agent_dist = jnp.full(self.num_cells, 1 / (self.num_cells - self.num_goals)).at[goals].set(0.0)
        agents = jax.random.choice(_rng, self.num_cells, (self.num_agents, ), p=agent_dist)
        agents_pos = jax.vmap(lambda x, shape: jnp.stack(jnp.unravel_index(x, shape)), in_axes=(0, None))(agents, (self.dim, self.dim))

        state = EnvState(
            agent=agents_pos,
            agents_term=jnp.zeros(self.num_agents, dtype=bool),
            goal=goals_pos,
            goals_term=jnp.zeros(self.num_goals, dtype=bool),
            time=0
        )
        
        goals_and_term = jnp.concatenate((state.goal, state.goals_term.reshape(-1, 1)), axis=1)
        flat_goals = goals_and_term.flatten()
        obs = jax.vmap(lambda pos: jnp.concatenate((pos, flat_goals)))(state.agent)

        return obs, state

    def _handle_actions(self, pos, action):
        step = jnp.int32(action == 0) * jnp.array([0, -1], jnp.int32) \
             + jnp.int32(action == 1) * jnp.array([0, 1], jnp.int32) \
             + jnp.int32(action == 2) * jnp.array([1, 0], jnp.int32) \
             + jnp.int32(action == 3) * jnp.array([-1, 0], jnp.int32)
        
        pos = pos + step

        return jnp.clip(pos, 0, self.dim - 1)

    def _handle_reward(
        self, pos, goals, goals_term, idx
    ):
        eq_fn = jax.vmap(lambda pos, goal, term: (pos == goal).all() * ~term, in_axes=(None, 0, 0))
        scores = eq_fn(pos[idx], goals, goals_term)

        overlap = jax.vmap(lambda pos1, pos2: (pos1 == pos2).all(), in_axes=(None, 0))(pos[idx], pos).sum() - 1
        score_coeff = jax.lax.select(
            idx == self.num_agents - 1,
            1.0, 
            jax.lax.select(
                overlap > 0,
                1 / overlap,
                1.0
            )
        )

        reward = scores.sum() * score_coeff
        return reward, scores


    def step_env(
        self, 
        rng, 
        state, 
        action,
        params
    ):
        # --- Update agent positions ---
        pos = jax.vmap(self._handle_actions)(state.agent, action)
        state = state.replace(agent=pos)

        # --- Compute utility ---
        agent_rewards, goal_term = jax.vmap(self._handle_reward, in_axes=(None, None, None, 0))(state.agent, state.goal, state.goals_term, jnp.arange(self.num_agents))

        agent_term = goal_term.max(axis=1)
        goal_term = goal_term.max(axis=0)
        team_rewards = agent_rewards[:-1].sum()
        adv_reward = agent_rewards[-1]
        
        reward = jax.lax.select(adv_reward > 0, jnp.full(self.num_agents, -1, dtype=float).at[-1].set(1), jnp.full(self.num_agents, team_rewards, dtype=float).at[-1].set(-team_rewards))

        # --- Update state ---
        state = EnvState(
            goal=state.goal,
            agent=pos,
            agents_term=agent_term,
            goals_term=goal_term,
            time=state.time + 1
        )

        # --- Calculate terminations ---
        done = jnp.logical_or(
            state.time > self.max_time,
            jnp.logical_or(
                state.agents_term[-1],
                state.goals_term.all(),
            )
        )

        # --- Calculate observations ---
        goals_and_term = jnp.concatenate((state.goal, state.goals_term.reshape(-1, 1)), axis=1)
        flat_goals = goals_and_term.flatten()
        obs = jax.vmap(lambda pos: jnp.concatenate((pos, flat_goals)))(state.agent)

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward,
            done, 
            {}
        )
    
    @property
    def name(self) -> str:
        """Environment name."""
        return "MultiGrid-TeamAdv-v1"