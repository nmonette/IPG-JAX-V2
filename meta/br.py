import jax
import jax.numpy as jnp

from models.optim import projection_simplex_truncated

def make_br(args, rollout_fn):

    def _br_fn(rng, train_state):

        def _br_loop(carry, t):

            rng, train_state = carry

            # --- Collect rollouts ---
            rng, _rng = jax.random.split(rng)
            rollout_data, returns, lambda_ = rollout_fn(_rng, train_state)

            # --- Compute regularized REINFORCE ---
            def epoch(carry, t):
                rng, train_state, rollout_data = carry

                def minibatch(carry, data):
                    train_state = carry

                    @jax.value_and_grad
                    def loss_fn(params, data):

                        probs = train_state.apply_fn({"params": params}, data.obs[None, ...])
                        action_probs = jax.vmap(lambda probs, idx: probs[idx])(probs, data.action)
                        log_probs = jnp.log(action_probs + 1e-6)

                        gamma = jnp.cumprod(jnp.full(log_probs.shape[0], args.gamma)) / args.gamma
                        temp_lambda = jax.vmap(lambda obs, action: lambda_[jnp.ravel_multi_index(obs, args.obs_dims), action])(data.obs, data.action)
                        reward = data.reward - args.nu * temp_lambda

                        return -(gamma * reward * log_probs.cumsum(axis=1) * ~data.done).sum()
                    
                    loss, grad = loss_fn(train_state.params, data)
                    train_state = train_state.apply_gradients(grads=grad)

                    train_state = train_state.replace(
                        params = projection_simplex_truncated(train_state.params["params"], args.trunc_size)
                    ) 

                    return train_state, loss
                
                rng, _rng = jax.random.split(rng)
                perm = jax.random.permutation(_rng, args.num_rollouts)
                num_mini_batches = args.num_rollouts // args.batch_size
                get_fn = lambda x: (
                    x[perm, -1]
                    .reshape(num_mini_batches, -1, x.shape[1:])
                )

                data = jax.tree_util.tree_map(get_fn, rollout_data)
                
                train_state, losses = jax.lax.scan(minibatch, train_state, data)
                
                return (rng, train_state, rollout_data), losses
            
            (rng, train_state, _), losses = jax.lax.scan(epoch, (rng, train_state, rollout_data), jnp.arange(args.num_epochs))

            metrics = {
                "adv_br_return": returns.mean(),
                "adv_br_loss": losses.mean()
            }

            return (rng, train_state), metrics
        
        (_, train_state), metrics = jax.lax.scan(_br_loop, (rng, train_state), jnp.arange(args.num_steps))

        return train_state, metrics
            
            


