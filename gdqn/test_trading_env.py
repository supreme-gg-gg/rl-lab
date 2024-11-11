import gymnasium as gym
import envs 

def test_environment():

    env = gym.make('trading-v0')
    #env.time_cost_bps = 0

    Episodes=3
    obs = []

    for _ in range(Episodes):
        observation = env.reset()[0]
        terminated, truncated = False, False
        count = 0

        while not terminated and not truncated:
            action = env.action_space.sample() # random
            observation, reward, terminated, truncated, info = env.step(action)
            obs = obs + [observation]
            
            if terminated or truncated:
                print(reward)
                print(count)

            count += 1
        
        print("Episode finished after {} timesteps".format(count))

if __name__ == "__main__":
    test_environment()
