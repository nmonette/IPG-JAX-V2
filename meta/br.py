import jax
import jax.numpy as jnp

from models.optim import projection_simplex_truncated

def make_br(args, rollout_fn, obs_dims):

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
                        probs = jax.vmap(jax.vmap(lambda obs: train_state.apply_fn(params, obs.reshape(1, -1))))(data.obs).squeeze()
                        action_probs = jax.vmap(jax.vmap(lambda probs, idx: probs[idx]))(probs, data.action)
                        log_probs = jnp.log(action_probs + 1e-6)
                        gamma = jnp.cumprod(jnp.full(log_probs.shape[1], args.gamma)) / args.gamma
                        temp_lambda = jax.vmap(jax.vmap(lambda obs, action: lambda_[jnp.ravel_multi_index(obs, obs_dims[1:], mode="clip"), action]))(data.obs, data.action)
                        reward = data.reward - args.nu * temp_lambda
                        return -(gamma * reward * log_probs.cumsum(axis=1) * ~data.done.squeeze()).sum()
                    
                    loss, grad = loss_fn(train_state.params, data)
                    train_state = train_state.apply_gradients(grads=grad)
                    
                    train_state = train_state.replace(
                        params = jax.tree_util.tree_map(lambda x: projection_simplex_truncated(x, args.trunc_size), train_state.params)
                    ) 

                    return train_state, loss
                
                rng, _rng = jax.random.split(rng)
                perm = jax.random.permutation(_rng, args.num_rollouts)
                num_mini_batches = args.num_rollouts // args.batch_size
                get_fn = lambda x: (
                    x[perm, :, -1]
                )
                reshape_fn = lambda x: (
                    x.reshape(num_mini_batches, -1, *x.shape[1:])
                )
                
                data = jax.tree_util.tree_map(get_fn, rollout_data)
                data = jax.tree_util.tree_map(reshape_fn, data)
                train_state, losses = jax.lax.scan(minibatch, train_state, data)
                
                return (rng, train_state, rollout_data), losses
            
            (rng, adv_train_state, _), losses = jax.lax.scan(epoch, (rng, train_state.adv_train_state, rollout_data), jnp.arange(args.num_epochs))

            metrics = {
                "adv_br_return": returns[-1].mean(),
                "adv_br_loss": losses.mean()
            }

            return (rng, train_state.replace(adv_train_state = adv_train_state)), metrics
        
        (_, train_state), metrics = jax.lax.scan(_br_loop, (rng, train_state), jnp.arange(args.num_br_steps))

        return train_state, metrics
    
    return _br_fn
            
            


