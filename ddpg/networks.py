from replay_buffer import *
from config import *
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, actions_dim, upper_bound, hidden_0=CRITIC_HIDDEN_0, hidden_1 = CRITIC_HIDDEN_1, 
                 init_minval=INIT_MINVAL, init_maxval=INIT_MAXVAL):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim

        self.init_minval = init_minval
        self.init_maxval = init_maxval
        self.upper_bound = upper_bound

        self.fc1 = nn.Linear(self.obs_dim, self.hidden_0)
        self.fc2 = nn.Linear(self.hidden_0, self.hidden_1)
        self.fc3 = nn.Linear(self.hidden_1, self.actions_dim)
        self.fc3.weight.data.uniform_(self.init_minval, self.init_maxval) # prevent zero gradient
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) # the output ranges from -1 to 1
        return x * self.upper_bound # we multiply by upper bound if env requries it between e.g. -2 and 2

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(Critic, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.fc1 = nn.Linear(obs_dim + act_dim, self.hidden_0) # 2 refers to [next_states, target_actions]
        self.fc2 = nn.Linear(self.hidden_0, self.hidden_1)
        self.fc3 = nn.Linear(self.hidden_1, 1) 

    def forward(self, state, action):

        x = torch.relu(self.fc1(torch.concat([state, action], axis=1)))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x