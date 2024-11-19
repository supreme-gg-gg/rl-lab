import gymnasium as gym

def play_cartpole():
    env = gym.make('CartPole-v1', render_mode="human")
    observation, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()
        # Simple strategy: move the cart in the direction of the pole's tilt
        action = 0 if observation[2] < 0 else 1
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward

    env.close()
    print(f'Total reward: {total_reward}')

if __name__ == "__main__":
    play_cartpole()