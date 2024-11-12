# We will solve: pendulum & lunar lander (continous version)

import time
import wandb
from tqdm import tqdm

import gymnasium as gym
import numpy as np

import torch

from config import *
from replay_buffer import *
from networks import *
from agent import *

config = {
    "learning_rate_actor": ACTOR_LR,
    "learning_rate_critic": ACTOR_LR,
    "batch_size": BATCH_SIZE,
    "architecture": "DDPG",
    "infra": "MacOS",
    "env": ENV_NAME
}

wandb.init(
    project=f"ddpg_{ENV_NAME.lower()}",
    config=config,
)

env = gym.make(ENV_NAME)
agent = DDPGAgent(env)

scores = []

if PATH_LOAD is not None:
    print("loading weights")
    observation = env.reset()
    action = agent.actor(torch.tensor(observation[None, :], dtype=torch.float32))
    print(agent.noise)
    agent.critic(torch.tensor(observation[None, :], dtype=torch.float32), action)
    agent.target_critic(torch.tensor(observation[None, :], dtype=torch.float32), action)
    agent.load_models()
    print(agent.replay_buffer.buffer_counter)
    print(agent.replay_buffer.n_games)
    print(agent.noise)

# Main training loop
for i in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    states, _ = env.reset()
    done = False
    score = 0
    noise = np.zeros(agent.actions_dim)
    while not done:
        action = agent.get_action(states, noise)
        new_states, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        agent.replay_buffer.push(states, action, reward, new_states, done)
        agent.learn()
        states = new_states
        
    agent.replay_buffer.update_n_games()
    
    scores.append(score)
    
    wandb.log({'Game number': agent.replay_buffer.n_games, '# Episodes': agent.replay_buffer.buffer_counter, 
           "Average reward": round(np.mean(scores[-10:]), 2), \
                   "Time taken": round(time.time() - start_time, 2)})
    
    if (i + 1) % EVALUATION_FREQUENCY == 0:
        states, _ = env.reset()
        done = False
        score = 0
        noise = np.zeros(agent.actions_dim) 
        while not done:
            action = agent.get_action(states, noise, evaluation=True)
            new_states, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            states = new_states
        wandb.log({'Game number': agent.replay_buffer.n_games, 
                '# Episodes': agent.replay_buffer.buffer_counter, 
                'Evaluation score': score})
        evaluation = False

agent.save_models()
