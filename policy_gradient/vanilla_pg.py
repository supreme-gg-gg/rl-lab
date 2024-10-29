# Tutorial: https://spinningup.openai.com/en/latest/algorithms/vpg.html
# Note that this heavily resembles the DQN code we wrote before, 
# except that it has a different update() method and uses A instead of RTG

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return Categorical(logits=x) # when inferencing no need to call Categorical again

# If we think of the policy as an actor, value function is the "critic"
# This, thus, makes a simple actor-critic architecture
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class VPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-2, gamma=0.9, lam=0.95):
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam # lambda parameter for GAE

    def select_action(self, obs):
        dist = self.policy_net(torch.tensor(obs, dtype=torch.float32))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy
    
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
        entropies = torch.stack([t[5] for t in trajectories])

        values = self.value_net(obs).squeeze().detach().numpy()
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # UPDATE POLICY
        self.policy_optimizer.zero_grad()
        # See notes and simple_pg for loss fn, added entropy regularization
        policy_loss = -(log_probs * advantages).mean() - 0.01 * entropies.mean() 
        policy_loss.backward()
        self.policy_optimizer.step()

        # UPDATE VALUE FUNCTION
        self.value_optimizer.zero_grad()
        values = self.value_net(obs).squeeze() # Simple MSE loss
        value_loss = nn.functional.mse_loss(values, returns)
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, env: gym.Env, num_episodes: int):
        for e in range(num_episodes):
            obs, _ = env.reset()
            trajectories = []
            episode_reward = 0

            done = False
            while not done:
                action, log_prob, entropy = self.select_action(obs)
                next_state, reward, done, _, _ = env.step(action)

                trajectories.append((obs, action, reward, log_prob, done, entropy))
                episode_reward += reward
                obs = next_state
            
            self.update(trajectories)
            print(f"Episode {e+1}: Total Reward: {episode_reward}")

        torch.save(self.policy_net.state_dict(), "../models/vpg.pth")
    
    def load_model(self, state_dict_path="../models/vpg.pth"):
        self.policy_net.load_state_dict(torch.load(state_dict_path, weights_only=True))
        self.policy_net.eval()

    def inference(self, env: gym.Env, episode=10, render=True) -> list:
    
        episode_rewards = []  # List to store total rewards for each episode
        
        for _ in range(episode):

            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                env.render()
                # Convert state to tensor and pass it to the model to get the action
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.inference_mode():
                    action = self.policy_net(state_tensor).sample().item()  # Select best action
                
                state, reward, done, _, _ = env.step(action)  # Take action in the environment
                total_reward += reward
                
                if render:
                    env.render()  # Render the environment if specified

            episode_rewards.append(total_reward)
            print(f'Total Reward: {total_reward}')
        
        env.close()  # Close the environment
        
        # Plot the rewards per episode
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid()
        plt.show()
        
        return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Inference VPG Agent")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Mode to run the agent in: 'train' or 'inference'")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for training or inference")
    parser.add_argument("--render", action="store_true", help="Render the environment during inference")
    args = parser.parse_args()

    if args.mode == "train":
        env = gym.make("CartPole-v1")
        agent = VPGAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
        agent.train(env, num_episodes=args.episodes)
    elif args.mode == "inference":
        env = gym.make("CartPole-v1", render_mode="human")
        agent = VPGAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
        agent.load_model()
        agent.inference(env, episode=args.episodes, render=args.render)