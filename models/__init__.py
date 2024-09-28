import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from .direct import DirectPolicy

def create_train_state(args, rng, obs_dims, num_actions, num_agents, adv = False):

    if args.policy == "direct":
        policy = DirectPolicy(obs_dims, num_actions, num_agents)
        params = policy.init(rng, jnp.zeros((num_agents, *obs_dims[1:]), dtype=int))

        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.sgd(args.adv_lr if adv else args.team_lr)
        )

        return TrainState.create(
            apply_fn = policy.apply,
            params = params,
            tx = tx
        )

    else:
        raise NotImplemented("Policy parameterization not implemented")
    
    