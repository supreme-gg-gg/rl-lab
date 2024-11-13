import argparse
import time
import wandb
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
from collections import Counter

from config import *
from replay_buffer import *
from networks import *
from agent import *

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.inference:
        env = gym.make(ENV_NAME, render_mode="human")
        agent = DDPGAgent(env, device)
        agent.load_models()
        states, _ = env.reset()
        done = False
        score = 0.0

        while not done:
            actions = agent.get_action(states, evaluation=True)
            new_states, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            score += rewards if not done else 0
            states = new_states

        print(f"Inference score: {score}")
        return

    # Vectorized Environments Setup
    # env = gym.vector.make(ENV_NAME, num_envs=NUM_ENVS)  # Use multiple environmentsA
    single_env = gym.make(ENV_NAME)
    agent = DDPGAgent(single_env, device)
    
    if ENABLE_WANDB:
        config = {
            "learning_rate_actor": ACTOR_LR,
            "learning_rate_critic": CRITIC_LR,
            "batch_size": BATCH_SIZE,
            "architecture": "DDPG",
            "infra": "MacOS",
            "env": ENV_NAME
        }

        wandb.init(
            project=f"ddpg_{ENV_NAME.lower()}",
            config=config,
        )
    
    scores = []

    for i in tqdm(range(MAX_GAMES)):
        start_time = time.time()
        states, _ = single_env.reset()
        done = False
        score = 0.0
        # states, _ = env.reset()  # Vectorized version
        # dones = np.array([False] * NUM_ENVS)  # Vectorized version
        # score = np.zeros(NUM_ENVS)  # Vectorized version

        timestep = 0
        max_timesteps = 1000  # Define a maximum number of timesteps per episode

        while not done and timestep < max_timesteps:  # Add timestep condition
            actions = agent.get_action(states)  # Will return single action
            new_states, rewards, terminated, truncated, _ = single_env.step(actions)
            done = terminated or truncated  # Single environment done flag
            score += rewards if not done else 0  # Single environment score update

            # Store experience in replay buffer (single environment version)
            agent.replay_buffer.push(states, actions, rewards, new_states, done)

            states = new_states

            if agent.replay_buffer.buffer_counter > BATCH_SIZE:
                critic_loss, actor_loss, q_value = agent.learn()

            timestep += 1  # Increment timestep

        scores.append(score)  # Changed from extend to append for single score

        if ENABLE_WANDB:
            wandb.log({
                'Game number': i + 1,
                'Average reward (last 10 games)': np.mean(scores[-10:]),
                'Time taken': round(time.time() - start_time, 2),
                'Critic Loss': critic_loss,
                'Actor Loss': actor_loss,
                'Average Q Value': q_value
            })


        if (i + 1) % SAVE_FREQUENCY == 0:
            print("saving...")
            agent.save_models()
            print("saved")

    agent.save_models()

    '''
    artifact = wandb.Artifact(name="model_saved", type="model")
    artifact.add_file("../models/actor.pth")
    artifact.add_file("../models/critic.pth")
    artifact.add_file("../models/target_critic.pth")
    wandb.log_artifact(artifact)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG Training and Inference")
    parser.add_argument('--inference', action='store_true', help='Run inference instead of training')
    args = parser.parse_args()
    main(args)