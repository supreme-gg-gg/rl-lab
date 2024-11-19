# tutorial: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

import argparse
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import time, random
import numpy as np
import gymnasium as gym
import wandb

from Agent import Agent
from config import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Inference mode")
    parser.add_argument("--exp-name", type=str, default="ppo", help="Name of the experiment")
    parser.add_argument("--env-name", type=str, default="Acrobot-v1", help="Name of the environment")
    parser.add_argument("--total_timesteps", type=int, default=25000, help="Number of time steps to train the model")
    parser.add_argument("--track", action="store_true", help="Track the experiment using WandB")
    parser.add_argument("--capture-video", action="store_true", help="Capture video of the environment")
    args = parser.parse_args()
    return args

## PPO deals with vector environments
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, 
                                               f"./videos/{run_name}", 
                                               episode_trigger=lambda x: x % 100 == 0)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.inference:
        envs = gym.make(args.env_name, render_mode="human")
        agent = Agent(envs, save_path=SAVE_PATH, load_path=LOAD_PATH).to(device)
        obs, _ = envs.reset()
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = terminated or truncated
            envs.render()
        envs.close()
        exit()

    run = f"{args.exp_name}_{args.env_name}_{int(time.time())}"

    if args.track:
        wandb.init(
            project="ppo",
            sync_tensorboard=True,
            name=run,
            monitor_gym=True,
            save_code=True,
            config=vars(args)  # Track hyperparameters
        )

    writer = SummaryWriter(f"runs/{run}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # 1. Vector environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, SEED + i, i, args.capture_video, run) 
        for i in range(NUM_ENVS)]
    )

    agent = Agent(envs, save_path=SAVE_PATH, load_path=LOAD_PATH).to(device)

    # 3. Adam epsilon
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

    global_step = 0
    batch_size = NUM_ENVS * NUM_STEPS
    mini_batch_size = batch_size // NUM_MINIBATCHES
    start_time  = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    num_updates = args.total_timesteps // batch_size

    for update in range(1, num_updates + 1):

        # 4. Learning rate annealing
        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = LEARNING_RATE * frac
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS # vector env
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad(): # no need gradient for rollout
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.from_numpy(np.logical_or(terminated, truncated).astype(np.float32)).to(device)

            if 'final_info' in info:
                for idx, item in enumerate(info['final_info']):
                    if item is not None and 'episode' in item:
                        episode = item['episode']
                        reward = episode['r'][0]
                        length = episode['l'][0]
                        print(f"Environment {idx}: Episodic Reward: {reward}, Length: {length}")
                        writer.add_scalar(f"charts/episodic_reward_env_{idx}", reward, global_step)
                        writer.add_scalar(f"charts/episodic_length_env_{idx}", length, global_step)
            
        # bootstrap reward if not done
        # 5. General Advantage Estimation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if GAE:
                advantges = torch.zeros_like(rewards).to(device)
                last_gae_lam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = rewards[t] + GANMA * next_values * next_non_terminal - values[t]
                    advantges[t] = last_gae_lam = delta + GANMA * LAMBDA * next_non_terminal * last_gae_lam
                returns = advantges + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    returns[t] = rewards[t] + GANMA * next_values * next_non_terminal
                advantges = returns - values
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantges.reshape(-1)
        b_values = values.reshape(-1)

        # update the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            # 6. Minibatch update
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]
                _, new_log_probs, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                log_ratio = new_log_probs - b_log_probs[mb_inds]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_EPSILON).float().mean()]
                
                mb_advantages = b_advantages[mb_inds]

                # 7. Advantage normalization
                if NORM_ADVANTAGES:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 8. Clipped objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # equivalent to min of positives

                # 9. Value loss clipping
                new_value = new_value.view(-1)
                v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_value - b_values[mb_inds], 
                    -CLIP_EPSILON, 
                    CLIP_EPSILON
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # 10. Entropy loss
                entropy_loss = entropy.mean() # entropy is maximized to encourage exploration
                loss = pg_loss - ENTROPY_COEFFICIENT * entropy_loss + VALUE_LOSS_COEFFICIENT * v_loss
                
                optimizer.zero_grad()
                loss.backward()

                # 11. Global gradient clipping
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            
            # 12. Early stopping
            if TARGET_KL is not None and approx_kl > TARGET_KL:
                break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    agent.save_model()

    if args.track:
        wandb.finish()