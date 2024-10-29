# Key Differences Between DQN and Policy Gradient Methods

## 1. Exploration Strategy: Epsilon-Greedy vs. Stochastic Policy

- **DQN (Q-Learning)**:
  - Uses **epsilon-greedy policy** to balance exploration and exploitation.
  - Chooses a random action with probability `ε`, otherwise selects the best-known action.
  - `ε` is decayed over time.
- **Policy Gradient (PG)**:
  - **No epsilon-greedy**; uses a **stochastic policy** directly (sampling actions from a probability distribution).
  - Encourages **exploration naturally** by sampling, without explicit randomness like epsilon.

## 2. Experience Replay vs. Trajectory Rollouts

- **DQN**:
  - Uses **experience replay**: stores past interactions (state, action, reward, next state) in a replay buffer.
  - Samples mini-batches of experiences, breaking correlations between consecutive actions.
- **PG**:
  - Uses **trajectory rollouts** (entire sequences of state-action-reward) per episode.
  - Updates policy based on collected trajectories, not sampled mini-batches.
  - **No experience replay** since PG methods rely on sequential, episodic updates.

## 3. Update Targets

- **DQN**:
  - Optimizes for the **Q-value** of the best action, minimizing the error in estimating `Q(s, a)`.
- **PG**:
  - Directly optimizes for **expected return** by adjusting the probability of actions that yield higher rewards.

## 4. Value Function

- **DQN**:
  - Learns the Q-values for state-action pairs.
- **PG**:
  - May learn a **value function** (Advantage Actor-Critic, A2C) but focuses on directly improving the policy.

---

# Double DQN vs. Policy Gradient (PG) – Key Differences

## 1. Use of Double Networks

- **Double DQN**:

  - Uses **two Q-networks** to mitigate overestimation bias in Q-learning:
    1. **Online network**: Selects the action with the highest Q-value.
    2. **Target network**: Provides the Q-value estimate for the chosen action.
  - Helps provide more stable Q-value estimates by decoupling action selection from evaluation.

- **Policy Gradient (PG)**:
  - Generally **does not use two networks** since it directly optimizes the policy.
  - No Q-value overestimation issue since PG does not estimate Q-values explicitly.
  - Some advanced PG variants (e.g., Actor-Critic) use two networks:
    - **Policy network (actor)**: Generates actions.
    - **Value network (critic)**: Estimates the value function to reduce variance in policy updates, but this differs fundamentally from Double DQN.

## 2. Avoiding Overestimation

- **Double DQN**:
  - Specifically designed to reduce the **overestimation bias** common in Q-learning by double-checking Q-value estimates.
- **PG**:
  - Optimizes action probabilities directly, so **overestimation bias is not an issue**.
  - Uses a value function (or advantage function) to guide updates, not to select actions based on Q-values.

## 3. Objective: Value Estimation vs. Direct Policy Optimization

- **Double DQN**:
  - Aims to estimate accurate Q-values while controlling overestimation, indirectly improving the policy by refining Q-value accuracy.
- **PG**:
  - Directly optimizes the policy by increasing the probability of higher-reward actions, focusing on **expected return** rather than value approximation.
