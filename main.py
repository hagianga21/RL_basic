import torch
import numpy as np
import random
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from DQN.dqn import Mario
from Common.utils import SkipFrame, GrayScaleObservation, ResizeObservation
from DQN.config import config_dqn
#Reproduce
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# Limit the action-space to  0. walk right  1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

mario = Mario(env, config_dqn)
mario.train(episodes = 300)