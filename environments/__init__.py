from .matrix import AdvMatrix, ENV_CONFIG as MATRIX_CONFIG
from .multigrid import AdvMultiGrid, ENV_CONFIG as GRID_CONFIG

def get_env(args):
    if args.env == "matrix":
        env = AdvMatrix(**MATRIX_CONFIG)

        obs_dims = tuple(i.n for i in env.observation_space().spaces)
        num_actions = env.num_actions
        num_agents = env.num_agents

        return env, obs_dims, num_actions, num_agents

    elif args.env == "multigrid":
        env = AdvMultiGrid(**GRID_CONFIG)
        
        obs_dims = tuple(i.n for i in env.observation_space().spaces)
        num_actions = env.num_actions
        num_agents = env.num_agents

        return env, obs_dims, num_actions, num_agents
    
    else:
        raise NotImplemented("Environment not implemented")