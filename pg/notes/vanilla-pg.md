# Vanilla Policy Gradient Summary

## Key Ideas

- **Policy Gradient**: Directly parameterizes the policy and optimizes it using gradient ascent on expected rewards.
- **Stochastic Policies**: Uses a distribution over actions, allowing exploration. We also use entropy regularisation.
- **Generalized Advantage Estimation (GAE)**: Balances bias and variance in advantage estimates.

## Core Components

1. **Policy Network**: A neural network that outputs action probabilities.
2. **Value Network**: A neural network that estimates the value of states.
3. **Trajectory Storage**: Stores experiences as tuples of (state, action, reward, log probability, done).

## Key Equations

### 1. Advantage Estimation

The advantage function is calculated using Generalized Advantage Estimation (GAE):

$$
A_t^{\text{GAE}} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots
$$

where the TD error $\delta_t$ is defined as:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 2. Returns Calculation

Returns are computed as:

$$
R_t = A_t + V(s_t)
$$

### 3. Policy Loss

The loss for updating the policy network is given by:

$$
L_{\text{policy}} = -\mathbb{E}[\log \pi(a_t | s_t) A_t]
$$

This is an approximation of the real policy gradient, which is given by:

$$
\nabla J(\theta) \approx \mathbb{E}\left[\nabla \log \pi(a_t | s_t; \theta) A_t\right]
$$

### 4. Value Loss

The loss for updating the value network is computed using Mean Squared Error (MSE):

$$
L_{\text{value}} = \mathbb{E}\left[(V(s_t) - R_t)^2\right]
$$

## Training Loop Overview

1. Collect trajectories from the environment.
2. Calculate advantages and returns using GAE.
3. Update the policy and value networks based on computed losses.
4. Repeat for a specified number of episodes.

## Conclusion

The vanilla Policy Gradient method provides a straightforward approach to optimize policies directly. By incorporating GAE, the method effectively reduces variance while maintaining bias at manageable levels, leading to more stable training.
