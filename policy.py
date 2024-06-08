# Implement a parametric policy, with parameters θ. 
# We recommend that you use PyTorch and a torch.nn.Module as the policy. 
# The module takes as input the state of the environment, and produces an action as output. 
# We suggest that it contains a single hidden layer of 128 neuros.
import torch
# The neural network module of pytorch
import torch.nn as nn
# The functional interface of pytorch
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        # Define the hidden layer
        self.hidden = nn.Linear(state_dim, hidden_dim)
        # Define the output layer
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # Apply the hidden layer
        x = self.hidden(state)
        # Apply the activation function
        actions = torch.tanh(self.output(x))
        # Return the action
        return actions
    
