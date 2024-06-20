import torch
# The neural network module of pytorch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        # Define the hidden layer
        self.hidden = nn.Linear(state_dim, hidden_dim)
        # Define the output layer
        self.output = nn.Linear(hidden_dim, action_dim)

    # Forward pass of the neural network
    def forward(self, state):
       x = torch.relu(self.hidden(state))
       return torch.tanh(self.output(x))

        

    
