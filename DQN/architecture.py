import torch.nn as nn
class Agent_Net(nn.Module):
    def __init__(self, config):
        super(Agent_Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['num_inputs'], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, config['num_actions'])
        )
        
    def forward(self, x):
        return self.layers(x)
