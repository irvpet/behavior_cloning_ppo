import torch
import torch.nn as nn
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, lr):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

        self.optim = optim.Adam(params=self.parameters(), lr=lr)
        self.device = 'cuda:0'
        self.to(self.device)


    def forward(self, obs):
        state = torch.tensor(obs, dtype=torch.float).to(self.device)
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


