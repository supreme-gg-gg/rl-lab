# RL Lab

This repository contains the code for various reinforcement learning algorithms implemented in PyTorch. The code is mostly my personal notes and experiments with these algorithms for education purpose. The code is not optimized, nor intended, for production use.

The algorithms are trained on various **Farama Gymansium environments**. The code is written in a modular way so that it can be easily extended to other environments. Some trained weights are also provided in the `models` directory for reference. I did not compare against openai baselines but feel free to optimize the code and compare the results.

> The repository is being updated regularly as I explore more algorithms.

Learn more about gymnasium here: [Farama Gymnasium](https://gymnasium.farama.org)

## Algorithms

There are only model-free DRL algorithms for now. **The list is not exhaustive, and I will keep adding more algorithms as I learn about them.**

| Q-Learning Algorithms              | Policy Gradient Algorithms                      |
| ---------------------------------- | ----------------------------------------------- |
| - [x] Deep Q-Network (DQN)         | - [x] Deep Deterministic Policy Gradient (DDPG) |
| - [x] Double Deep Q-Network (DDQN) | - [x] Proximal Policy Optimization (PPO)        |
|                                    | - [x] REINFORCE (vanilla policy gradient)       |

## Environments

## Environments

| Classic Control | Box2D            | Atari                  |
| --------------- | ---------------- | ---------------------- |
| CartPole-v1     | LunarLander-v2   | PongNoFrameskip-v4     |
| MountainCar-v0  | BipedalWalker-v3 | BreakoutNoFrameskip-v4 |
| Acrobot-v1      |                  |                        |
| Pendulum-v0     |                  |                        |

## Requirements

Using a `requirements.txt` will likely not work so I will not provide one. You should just install these core libraries used using your own preferred method. I use `conda` to install (most of) the libraries.

- `torch` (used for almost all models)
- `tensorflow` (only used in 1-2 models)
- `gymnasium` (for Gym classical control)
- `ale-py` (for Atari environments)
- `stable-baselines3` (wrapper for Gym environments)
- `tensorboard` (for logging)
- `wandb` (for logging)
- `scikit-learn`, `numpy`, `matplotlib`, `pandas`, `scipy`
- `yfinance` (for financial data API)

## Former Quantitative Trading Project

The `gdqn` and `deep_learning` folder contains the code for a former quantitative trading project. The project was an attempt to use reinforcement learning to train a trading agent to trade on the stock market. The agent and environment and data have already been built. The project is still in progress but I amm not sure when will it progress further.

### Financial Optimization Notes

The `optimization` folder contains some notes on financial optimization. Many people do experiments with using reinforcement learning in financial markets, so I believe it is a good idea to build up some basic knowledge regarding numerical optimization in finance.

> Again, the algorithms are not optimized and are for educational purposes only.

## Resources

I would like to give thanks to:

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) by Richard S. Sutton and Andrew G. Barto

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)

- [DeepMind X UCL](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)

- Various online textbooks [such as this one](https://gibberblot.github.io/rl-notes/intro/intro.html)

- Various blog articles and YouTube videos

## Research Papers

The following research papers were used as references for the implementation of the algorithms. This is **not an exhaustive list**, but it is a good starting point for anyone interested in learning more about reinforcement learning. I would keep adding to them from time to time.

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
