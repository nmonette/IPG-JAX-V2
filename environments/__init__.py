from .matrix import AdvMatrix, ENV_CONFIG as MATRIX_CONFIG

def get_env(args):
    if args.env == "matrix":
        env = AdvMatrix(MATRIX_CONFIG)

        obs_dims = tuple(i.n for i in env.observation_space().spaces)
        num_actions = env.num_actions
        num_agents = env.num_agents

        return env, obs_dims, num_actions, num_agents
    
    else:
        raise NotImplemented("Environment not implemented")