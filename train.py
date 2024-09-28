import jax
import jax.numpy as jnp
import tyro
import wandb

from dataclasses import dataclass

from meta import (
    make_rollout,
    make_br,
    make_reinforce,
    make_nash_gap,
    TrainState
)
from models import create_train_state
from environments import get_env

@dataclass
class Args:
    # Environment
    env: str = "matrix"
    num_steps: int = 1000
    num_br_steps: int = 100
    rollout_length: int = 8
    num_rollouts: int = 512

    # Model
    policy: str = "direct"

    # Optimiziation
    adv_lr: float = 1e-4
    team_lr: float = 1e-3
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    nu: float = 0.05
    trunc_size: float = 0.05

    # Experiment
    seed: int = 0
    batch_size: int = 32
    num_epochs: int = 4
    log_interval: int = 10
    
    # WandB
    project: str = "ipg-jax"
    entity: str = "nathanmonette1"
    group: str = "debug"
    log: bool = False


def make_train(args):

    def _train_fn(rng):

        env, obs_dims, num_actions, num_agents = get_env(args)

        rollout_fn = make_rollout(args, env, obs_dims, num_actions, num_agents)
        reinforce_fn = make_reinforce(args, rollout_fn)
        br_fn = make_br(args, rollout_fn)
        gap_fn = make_nash_gap(args, rollout_fn, br_fn, num_agents)

        rng, _rng = jax.random.split(rng)
        team_train_state = create_train_state(args, _rng, obs_dims, num_actions, num_agents - 1)
        rng, _rng = jax.random.split(rng)
        adv_train_state = create_train_state(args, _rng, obs_dims, num_actions, 1, True)

        train_state = TrainState(team_train_state, adv_train_state)
            
        def loop(carry, t):
            rng, train_state = carry

            rng, _rng = jax.random.split(rng)
            train_state, adv_metrics = br_fn(_rng, train_state)

            rng, _rng = jax.random.split(rng)
            train_state, team_metrics = reinforce_fn(_rng, train_state)

            metrics = {
                **team_metrics,
                "adv_br_loss": adv_metrics["adv_br_loss"][-1],
                "adv_return": adv_metrics["adv_br_return"][-1]
            }

            return (rng, train_state), (metrics, train_state)

        rng, _rng = jax.random.split(rng)
        (rng, train_state), (metrics, all_train_states) = jax.lax.scan(loop, (rng, train_state), xs=jnp.arange(args.num_steps))

        avg_train_states = all_train_states.replace(
            team_train_state = all_train_states.team_train_state.replace(
                params = jax.tree_util.tree_map(
                    lambda x: x.cumsum(axis=0) / jnp.ones(x.shape[0]).cumsum().reshape(args.num_steps, -1, 1, 1), all_train_states.team_train_state.params
                )
            ),
            adv_train_state = all_train_states.adv_train_state.replace(
                params = jax.tree_util.tree_map(
                    lambda x: x.cumsum(axis=0) / jnp.ones(x.shape[0]).cumsum().reshape(args.num_steps, -1, 1, 1), all_train_states.adv_train_state.params
                )
            )
        )

        ## NOTE: we should really do this every n steps instead haha
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, args.num_steps)
        gaps = jax.vmap(gap_fn)(_rng, avg_train_states)

        metrics["nash_gap"] = gaps

        return metrics
    
    return _train_fn


def main():

    args = tyro.cli(Args)

    if args.log:
        wandb.init(
            config=args,
            project=args.project,
            entity=args.entity,
            group=args.group,
            job_type="train",
        )

    rng = jax.random.key(args.seed)
    train_fn = jax.jit(make_train(args))

    metrics = train_fn(rng)

    if args.log:
        for step in range(args.num_steps):
            wandb.log(
                jax.tree_util.tree_map(lambda x: x[step], metrics)
            )
    else:
        print(metrics)


if __name__ == "__main__":
    main()