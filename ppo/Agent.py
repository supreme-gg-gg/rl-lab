import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch
import gymnasium as gym

# 2. Layer initialization
def layers_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, 
                 save_path="../models/cartpole/ppo_model.pth", 
                 load_path="../models/cartpole/ppo_model.pth"):
        super(Agent, self).__init__()
        self.envs = envs
        
        if isinstance(envs, gym.vector.VectorEnv):
            self.action_dim = envs.single_action_space.n
            self.obs_dim = np.array(envs.single_observation_space.shape).prod()
        else:
            self.action_dim = envs.action_space.n
            self.obs_dim = np.array(envs.observation_space.shape).prod()

        self.network = nn.Sequential(
            layers_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layers_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor = layers_init(nn.Linear(64, self.action_dim), std=0.01)
        self.critic = layers_init(nn.Linear(64, 1), std=1)
        self.save_path = save_path
        self.load_path = load_path

    def get_value(self, x):
        return self.critic(self.network(x))
    
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def save_model(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, self.save_path)

    def load_model(self):
        checkpoint = torch.load(self.load_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

class CNNAgent(nn.Module):
    def __init__(self, envs, 
                 save_path="../models/breakout/ppo_model.pth", 
                 load_path="../models/breakout/ppo_model.pth"):
        super(CNNAgent, self).__init__()
        self.envs = envs

        if isinstance(envs, gym.vector.VectorEnv):
            self.action_dim = envs.single_action_space.n
        else:
            self.action_dim = envs.action_space.n
         
        # 2. Shared networks for feature extraction and different heads for policy and value
        self.network = nn.Sequential(
            layers_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layers_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layers_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layers_init(nn.Linear(64*7*7, 512)),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            layers_init(nn.Linear(512, self.action_dim), std=0.01)
        )
        self.critic = nn.Sequential(
            layers_init(nn.Linear(512, 1), std=1.0)
        )
        self.save_path = save_path
        self.load_path = load_path
        
    def get_value(self, x):
        return self.critic(self.network(x / 255.0))
    
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0) # scaling input to [0, 1]
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def save_model(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, self.save_path)

    def load_model(self):
        checkpoint = torch.load(self.load_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

    