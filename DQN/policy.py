import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from DQN.architecture import Agent_Net

class DQNPolicy():
    def __init__(self, config):
        self.device = config['device']
        self.model = Agent_Net(config).to(self.device)
        self.target_model = Agent_Net(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = config['learning_rate'])

    def calc_loss_and_optimize(self, y_pred, y):
        self.model.train()
        loss = F.smooth_l1_loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_with_grad(self, nn_input):
        self.model.train()
        y_pred = self.model(nn_input)
        return y_pred
    
    def predict(self, nn_input, target):
        with torch.no_grad():
            if target:
                self.target_model.eval()
                y_pred = self.target_model(nn_input)
            else:
                self.model.eval()
                y_pred = self.model(nn_input)
        return y_pred
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, folder, filename):
        """
        Save the model
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model.state_dict(), filepath)
    
    def load_saved_model(self, folder, filename):
        """
        Load the saved model
        """
        filepath = os.path.join(folder, filename)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))