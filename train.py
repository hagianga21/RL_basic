#from DQN.dqn import DQNAgent
#from DQN.config import config_dqn
import numpy as np
import random 
import torch
import gym

if __name__=="__main__":
    #Reproduce results
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env_id = "CartPole-v0"
    env = gym.make(env_id)
