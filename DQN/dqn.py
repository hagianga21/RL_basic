from DQN.replayMemory import ReplayMemory
from DQN.policy import DQNPolicy
from Common.logger import TensorBoard_Record
import torch
from itertools import count
from collections import namedtuple
import random
import math

class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.policy = DQNPolicy(config)
        self.device = config['device']
        self.memory = ReplayMemory(config['memory_capacity'])
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.target_update = config['target_update']
        self.steps_done = 0
        self.i_episode = 0
        self.logger = TensorBoard_Record("tensorboard", "DQN")
    
    #def select_action(self, graph, avail_idx):
        

    #def optimize_model(self):
        
        
    #def run_episode(self, memory_initialization):
        
        
    #def train(self, epochs = 4000):