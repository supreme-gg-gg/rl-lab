# Tutorial: https://spinningup.openai.com/en/latest/algorithms/vpg.html
# Note that this heavily resembles the DQN code we wrote before, 
# except that it has a different update() method and uses A instead of RTG

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return Categorical(logits=x) # when inferencing no need to call Categorical again

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class VPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99, lam=0.95):
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam # lambda parameter for GAE

    def select_action(self, obs) -> Categorical:
        dist = self.policy_net(torch.tensor(obs, dtype=torch.float32))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    # Generalized Advantage Estimation (learn this later)
    # Note that it looks like RTG calculation since A replaces R lol
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
            delta = rewards[i] + self.gamma * next_value - values[i] # r(t) + gamma * V(t+1) - V(t)
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            next_value = values[i]
        
        returns = [adv + val for adv, val in zip(advantages, values)] # by definition
        return advantages, returns
    
    def update(self, trajectories):
        # NOTE: You can also do this with a Memory NamedTuple called Experience Replay as in DQN
        obs = torch.tensor([t[0] for t in trajectories], dtype=torch.float32)
        acts = torch.tensor([t[1] for t in trajectories], dtype=torch.int32)
        rewards = [t[2] for t in trajectories]
        log_probs = torch.stack([t[3] for t in trajectories])
        dones = [t[4] for t in trajectories]

        values = self.value_net(obs).squeeze().detach().numpy()
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # UPDATE POLICY
        self.policy_optimizer.zero_grad()
        policy_loss = -(log_probs * advantages).mean() # See notes and simple_pg
        policy_loss.backward()
        self.policy_optimizer.step()

        # UPDATE VALUE FUNCTION
        self.value_optimizer.zero_grad()
        values = self.value_net(obs).squeeze() # Simple MSE loss
        value_loss = nn.functional.mse_loss(values, returns) #
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, env: gym.Env, num_episodes: int):
        for e in range(num_episodes):
            obs, _ = env.reset()
            trajectories = []
            episode_reward = 0

            done = False
            while not done:
                action, log_prob = self.select_action(obs)
                next_state, reward, done, _, _ = env.step(action)

                trajectories.append((obs, action, reward, log_prob, done))
                episode_reward += reward
                obs = next_state
            
            self.update(trajectories)
            print(f"Episode {e+1}: Total Reward: {episode_reward}")

        torch.save(self.policy_net.state_dict(), "models/vpg.pth")
        torch.save(self.value_net.state_dict(), "models/vpg_values.pth")

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = VPGAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    agent.train(env, num_episodes=500)