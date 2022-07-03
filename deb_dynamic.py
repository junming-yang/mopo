from models_torch.transition_model import TransitionModel
import gym

if __name__ == "__main__":
    env_name = "Hopper-v3"
    env = gym.make(env_name)
    obs_space = env.observation_space
    action_space = env.action_space
    transition_model = TransitionModel(obs_space, action_space, env_name=env_name)
