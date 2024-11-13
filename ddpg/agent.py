import os
import torch
import torch.nn as nn
import numpy as np
from networks import *
from replay_buffer import *
from config import *
import wandb

class DDPGAgent():
    def __init__(self, env, device, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, max_size=BUFFER_CAPACITY, 
                 tau=TAU, path_save=PATH_SAVE, path_load=PATH_LOAD) -> None:
        
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env, max_size)
        self.obs_dim = env.observation_space.shape[0]
        self.actions_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.path_save = path_save
        self.path_load = path_load

        self.actor = Actor(obs_dim=self.obs_dim, actions_dim=self.actions_dim, upper_bound=self.upper_bound).to(device)
        self.critic = Critic(obs_dim=self.obs_dim, act_dim=self.actions_dim).to(device)
        self.target_actor = Actor(obs_dim=self.obs_dim, actions_dim=self.actions_dim, upper_bound=self.upper_bound).to(device)
        self.target_critic = Critic(obs_dim=self.obs_dim, act_dim=self.actions_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # self.target_actor_optimizer = torch.optim.Adam(self.target_actor.parameters(), lr=self.actor_lr)
        # self.target_critic_optimizer = torch.optim.Adam(self.target_critic.parameters(), lr=self.critic_lr)

        self.device = device

    def update_target_networks(self, tau):
        # Soft update for target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Soft update for target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # for just inference, only load and save actor and critic.pth but not targets
    def save_models(self):
        print('... saving models ...')
        torch.save(self.actor.state_dict(), os.path.join(self.path_save, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(self.path_save, 'critic.pth'))
        torch.save(self.target_actor.state_dict(), os.path.join(self.path_save, 'target_actor.pth'))
        torch.save(self.target_critic.state_dict(), os.path.join(self.path_save, 'target_critic.pth'))

    def load_models(self):
        print('... loading models ...')
        self.actor.load_state_dict(torch.load(os.path.join(self.path_load, 'actor.pth'), map_location=self.device, weights_only=True))
        self.critic.load_state_dict(torch.load(os.path.join(self.path_load, 'critic.pth'), map_location=self.device, weights_only=True))
        self.target_actor.load_state_dict(torch.load(os.path.join(self.path_load, 'target_actor.pth'), map_location=self.device, weights_only=True))
        self.target_critic.load_state_dict(torch.load(os.path.join(self.path_load, 'target_critic.pth'), map_location=self.device, weights_only=True))

    # exploration strategy for deterministic policy
    # simplest: add noise to the action returned by Agent
    @DeprecationWarning
    def _ornstein_uhlenbeck_process(self, x, theta=THETA, mu=0, dt=DT, std=0.2):
        """
        Ornstein–Uhlenbeck process
        """
        return x + theta * (mu-x) * dt + std * np.sqrt(dt) * np.random.normal(size=self.actions_dim)
    
    def get_action(self, observation: tuple, evaluation=False):

        observation = np.array(observation, dtype=np.float32)
        if observation.ndim == 1:
            observation = np.expand_dims(observation, axis=0)
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        if not evaluation:
            actions = self.actor.noisy_forward(state).detach().cpu().numpy()
        else:
            actions = self.actor.forward(state).detach().cpu().numpy()
        actions = np.clip(actions, self.lower_bound, self.upper_bound)

        return actions[0] # Box action shape requries a vector, Discrete -- scalar
    
    def learn(self):
        if self.replay_buffer.check_buffer_size() is False:
            return 0.0, 0.0, 0.0 # placeholders
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # almost exactly as how you train DQN
        with torch.no_grad(): # you don't want to train target network
            target_actions = self.target_actor(next_states)
            target_critic_values = self.target_critic(next_states, target_actions.detach()).squeeze(1)

        critic_value = self.critic(states, actions).squeeze(1)
        target = rewards + self.gamma * target_critic_values * (1-dones)
        critic_loss = nn.MSELoss()(critic_value, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # In short you want to take actions that lead to higher Q value
        # so max the Q(s,a) where a = pi(s)
        policy_actions = self.actor(states)
        actor_loss = -self.critic(states, policy_actions)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks(self.tau)

        return critic_loss.item(), actor_loss.item(), critic_value.mean().item()

        