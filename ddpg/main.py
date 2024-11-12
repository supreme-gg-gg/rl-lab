import argparse
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

def main(args):

    if args.inference:
        env = gym.make(ENV_NAME, render_mode="human")
        agent = DDPGAgent(env)

        if PATH_LOAD is not None:
            print("loading weights")
            agent.load_models()
        
        states, _ = env.reset()
        done = False
        score = 0
        noise = np.zeros(agent.actions_dim)
        while not done:
            action = agent.get_action(states, noise, evaluation=True)
            new_states, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            states = new_states
        print(f"Inference score: {score}")
        return
    
    env = gym.make(ENV_NAME)
    agent = DDPGAgent(env)
    
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

    scores = []

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

        # Log Q-value estimate on a batch of states and actions from the replay buffer
        if agent.replay_buffer.buffer_counter > agent.replay_buffer.batch_size:
            # Sample batch for a more representative Q-value
            batch_states, batch_actions, _, _, _ = agent.replay_buffer.sample()
            with torch.no_grad():
                q_values = agent.critic(torch.tensor(batch_states, dtype=torch.float32), torch.tensor(batch_actions, dtype=torch.float32)).mean().detach().item()
            wandb.log({"Average Q-value": q_values})

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
            
        if (i + 1) % SAVE_FREQUENCY == 0:
            print("saving...")
            agent.save_models()
            print("saved")

    agent.save_models()

    # Create and log artifact
    artifact = wandb.Artifact(name="model_saved", type="model")
    artifact.add_file("../models/actor.pth")
    artifact.add_file("../models/critic.pth")
    artifact.add_file("../models/target_critic.pth")
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG Training and Inference")
    parser.add_argument('--inference', action='store_true', help='Run inference instead of training')
    args = parser.parse_args()
    main(args)
