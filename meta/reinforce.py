import jax
import jax.numpy as jnp

from models.optim import projection_simplex_truncated

def make_reinforce(args, rollout_fn):
    
    def reinforce(rng, train_state):
        # --- Collect rollouts ---
        rng, _rng = jax.random.split(rng)
        rollout_data, returns, _ = rollout_fn(_rng, train_state)

        # --- Compute regularized REINFORCE ---
        def epoch(carry, t):
            rng, train_state, rollout_data = carry

            def minibatch(carry, data):
                train_state = carry

                @jax.value_and_grad
                def loss_fn(params, data):
        
                    probs = jax.vmap(jax.vmap(lambda obs: train_state.apply_fn(params, obs)))(data.obs)

                    def inner_fn(data, probs):
                        action_probs = jax.vmap(jax.vmap(lambda probs, idx: probs[idx]))(probs, data.action)
                        log_probs = jnp.log(action_probs + 1e-6)
                        gamma = jnp.cumprod(jnp.full(log_probs.shape[1], args.gamma)) / args.gamma

                        return -(gamma * data.reward * log_probs.cumsum(axis=1) * data.valid_mask.squeeze()).sum()

                    return jax.vmap(inner_fn)(data, probs).mean()

                
                loss, grad = loss_fn(train_state.params, data)
                train_state = train_state.apply_gradients(grads=grad)
                
                if args.policy == "direct":
                    train_state = train_state.replace(
                        params = jax.tree_util.tree_map(lambda x: projection_simplex_truncated(x, args.trunc_size), train_state.params)
                    ) 

                return train_state, loss
            
            rng, _rng = jax.random.split(rng)
            perm = jax.random.permutation(_rng, args.num_rollouts)
            num_mini_batches = args.num_rollouts // args.batch_size
            get_fn = lambda x: (
                x[perm, :, :-1]
            )
            reshape_fn = lambda x: (
                x.reshape(num_mini_batches, -1, *x.shape[1:])
            )
            
            data = jax.tree_util.tree_map(get_fn, rollout_data)
            data = jax.tree_util.tree_map(reshape_fn, data)
            train_state, losses = jax.lax.scan(minibatch, train_state, data)
            
            return (rng, train_state, rollout_data), losses
        
        (rng, team_train_state, _), losses = jax.lax.scan(epoch, (rng, train_state.team_train_state, rollout_data), jnp.arange(args.num_epochs))

        metrics = {
            "team_return": returns[0],
            "team_loss": losses.mean()
        }

        return train_state.replace(team_train_state=team_train_state), metrics

    return reinforce