import gymnasium as gym

# Create the LunarLander environment
env = gym.make("LunarLander-v2", render_mode="human")

observation, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    observation, reward, done, truncated, info = env.step(action)
    if done:
        observation, info = env.reset()

env.close()