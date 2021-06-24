import torch
import numpy as np
from DQN.architecture import MarioNet
from Common.logger import TensorBoard_Record
from Common.logger2 import MetricLogger
from replayMemory import ReplayMemory
from pathlib import Path

class Mario:
    def __init__(self, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(self.device)
            
        self.exploration_rate = config['exploration_rate']
        self.exploration_rate_decay = config['exploration_rate_decay']
        self.exploration_rate_min = config['exploration_rate_min']
        self.curr_step = 0

        self.memory = ReplayMemory(config['memory_capacity'])
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['learning_rate'])
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 100  # min. experiences before training
        self.learn_every = 3
        self.sync_every = 1e4
        #self.logger = TensorBoard_Record("Tensorboard", "DQN")
        self.logger = MetricLogger(Path("log"))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state.__array__()
            state = torch.tensor(state).to(self.device)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        state = torch.tensor(state.__array__())
        next_state = torch.tensor(next_state.__array__())
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        self.memory.push(state, next_state, action, reward, done)
    
    def recall(self):
        batch = self.memory.sample(self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state.to(self.device), next_state.to(self.device), action.squeeze().to(self.device), reward.squeeze().to(self.device), done.squeeze().to(self.device)

    def td_estimate(self, state, action):
        # Q_online(s,a)
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)
    
    def train(self, episodes):
        total_reward = []
        for e in range(episodes):
            episode_reward = 0
            state = self.env.reset()
            # Play the game!
            while True:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward = episode_reward + reward
                self.cache(state, next_state, action, reward, done)
                q, loss = self.learn()
                self.logger.log_step(reward, loss, q)
                state = next_state
                if done or info["flag_get"]:
                    print("Episode: %d - Reward: %d" %(e, episode_reward))
                    total_reward.append(episode_reward)
                    break
            self.logger.log_episode()
            if e % 20 == 0:
                self.logger.record(episode=e, epsilon=self.exploration_rate, step=self.curr_step)

