import jax
import jax.numpy as jnp
from flax.struct import dataclass

from math import prod

@dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray

def make_rollout(args, env, obs_dims, num_actions, num_agents):

    def collect_rollouts(
        rng, 
        train_state,
    ):

        def _env_loop(carry, t):
            rng, train_state, obs, env_state, returns, valid_mask, gamma, lambda_ = carry

            actions = train_state.get_actions(rng, obs)
            rng, _rng = jax.random.split(rng)
            next_obs, next_state, reward, done, info = env.step(
                _rng, env_state, actions, env.default_params
            )

            transition = Transition(
                obs=obs,
                action=actions,
                reward=reward,
                next_obs=next_obs, 
                done=jnp.full((num_agents, 1), done), 
            )

            carry = (
                rng, 
                train_state,
                next_obs, 
                next_state, 
                returns + gamma * reward * valid_mask, 
                valid_mask * (1 - done),
                gamma * args.gamma,
                lambda_.at[jnp.ravel_multi_index(obs[-1], (lambda_.shape[0], ), mode="clip"), actions[-1]].add(gamma)
            )

            return carry, transition
        
        def scan_fn(rng, train_state):
        
            # --- Initialize environment ---
            rng, _rng = jax.random.split(rng)
            init_obs, init_state = env.reset(_rng, env.default_params) # vmap reset env

            # --- Initialize state-action visitation estimator ---
            lambda_ = jnp.zeros((prod(obs_dims), num_actions))
            
            carry_out, transitions = jax.lax.scan(
                _env_loop, (
                    rng, 
                    train_state,
                    init_obs, 
                    init_state,
                    jnp.zeros(num_agents),
                    jnp.float32(1.0),
                    args.gamma, 
                    lambda_
                ), length=args.rollout_length)
            
            # --- Add final value onto end of rollouts ---
            (
                rng, 
                train_state, 
                _, _,
                returns, 
                _, _,
                lambda_
            ) = carry_out

            return transitions, returns, lambda_
        
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(rng, args.num_rollouts)
        transitions, returns, lambda_ =  jax.vmap(scan_fn, in_axes=(0, None))(_rng, train_state)
        return transitions, returns.mean(axis=0), lambda_.mean(axis=0)
    
    return collect_rollouts