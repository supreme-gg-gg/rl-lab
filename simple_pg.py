# Tutorial: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

def mlp(sizes: list, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        # act is reference to activation function, act() creates an instance and add to list
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers) # unpack list of layers into individual elements

def inference(path='models/simple_policy_model.pth', env_name='CartPole-v1', hidden_sizes=[32]):
    env = gym.make(env_name, render_mode="human")
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    logits_net.load_state_dict(torch.load(path, weights_only=True))
    logits_net.eval()
    total_reward = 0
    with torch.inference_mode():
        obs, _ = env.reset()
        done = False
        iteration = 0
        while not done:
            env.render()
            obs = torch.as_tensor(obs, dtype=torch.float32)
            logits = logits_net(obs)
            action = Categorical(logits=logits).sample().item()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            iteration += 1
            if iteration % 100 == 0:  # Print total reward every 10 iterations
                print(f'Total Reward: {total_reward}')

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):
    
    # setup the gym env
    if render:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, redner_mode="human")
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Make the policy network (feedforward)
    # NOTE: sizes arg is a list of sizes at different layers!
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (output int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, from the right data, is the policy gradient
    # the obs and act you should plug in are the dataset collected according to currenct policy
    def compute_loss(obs, act, weights):

        '''
        THIS IS NOT LOSS FUNCTION IN THE TRADITIONAL SENSE!
        In policy gradients, we only care about average return.
        The loss function (decreasing) means nothing.
        '''

        logp = get_policy(obs).log_prob(act) 
        # when working with Normal distrib do log_prob(act).sum(axis=-1)
        return -(logp * weights).mean()
    
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # Training policy
    def train_one_epoch():

        # these lists are just for logging
        batch_obs = []
        batch_acts = []
        batch_weights = [] # R(tau) weighting in PG (different update methods)
        batch_rets = []
        batch_lens = []

        obs, _ = env.reset()
        done = False
        ep_rews = [] # rewards accured throughout episode

        finished_rendering_this_epoch = False

        # 1. Experience collection: agent act some episodes in env using most recent olicy
        while True:

            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy()) # save obs
            
            # act in env according to policy
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # record info when episode over
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # DEPRECATED: weight for each logprob(a|s) is R(tau) 
                # batch_weights += [ep_ret] * ep_len

                # NOTE: if we use reward to go:
                batch_weights += list(reward_to_go(ep_rews))

                obs, _ = env.reset()
                done, ep_rews = False, [] # reset episode wide var

                finished_rendering_this_epoch = True # won't render this episode again

                # end experience loop if we have enough of it
                if (len(batch_obs) > batch_size):
                    break

        # 2. Take a single policy gradient update step (good old PyTroch train loop)
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                    act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                    weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()

        return batch_loss, batch_rets, batch_lens

    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs
    
    # TRAINING LOOP 
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        
    # Save the state dictionary of the model
    torch.save(logits_net.state_dict(), 'models/simple_policy_model.pth')
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train')
    parser.add_argument('--model_path', type=str, default='models/simple_policy_model.pth')
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    if args.mode == 'train':
        train(env_name=args.env_name, render=args.render, lr=args.lr)
    else:
        inference(path=args.model_path, env_name=args.env_name)
    args = parser.parse_args()