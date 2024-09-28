import jax
import jax.numpy as jnp
import tyro
import wandb

from dataclasses import dataclass

from meta import (
    make_rollout,
    make_br
)
from models import create_train_state
from environments import get_env

@dataclass
class Args:
    # Environment
    env: str = "matrix"
    num_steps: int = 1000
    rollout_length: int = 10
    num_rollouts: 512

    # Model
    policy: str = "direct"

    # Optimiziation
    adv_lr: float = 1e-5
    team_lr: float = 1e-4
    max_grad_norm: float = 0.5

    # Experiment
    seed: int = 0
    batch_size: int = 32
    log_interval: int = 10
    
    # WandB
    project: str = "ipg-jax"
    entity: str = "nathanmonette1"
    group: str = "debug"



def make_train(args):

    def _train_fn(rng):

        env, obs_dims, num_actions, num_agents = get_env(args)

        rollout_fn = make_rollout(args, env, obs_dims, num_actions)
        br_fn = make_br(args, rollout_fn)

        rng, _rng = jax.random.split(rng)
        team_train_state = create_train_state(args, _rng, obs_dims, num_actions, num_agents)
        rng, _rng = jax.random.split(rng)
        adv_train_state = create_train_state(args, _rng, obs_dims, num_actions, 1)

        def _loop_fn(carry, t):

            rng, team_train_state, adv_train_state = carry
            
            adv_train_state, metrics = br_fn(rng, adv_train_state)
            
            return (rng, team_train_state, adv_train_state), metrics
        
        for _ in range(args.num_steps // args.log_interval):

            rng, team_train_state, adv_train_state, metrics = jax.lax.scan(_loop_fn, (rng, team_train_state, adv_train_state), jnp.arange(args.log_interval))

            for step in range(args.log_interval):
                wandb.log(
                    jax.tree_util.tree_map(lambda x: x[step], metrics)
                )
        
        return team_train_state, adv_train_state
    
    return _train_fn


def main():

    args = tyro.cli(Args)

    wandb.init(
        config=args,
        project=args.project,
        entity=args.entity,
        group=args.group,
        job_type="train",
    )

    rng = jax.random.key(args.seed)
    train_fn = jax.jit(make_train(args))

    team_train_state, adv_train_state = train_fn(rng)



        