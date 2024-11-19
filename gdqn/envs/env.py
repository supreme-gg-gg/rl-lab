import logging
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gdqn.utils import get_data

logging.basicConfig(filename="training.log", level=logging.INFO)

'''
The environment will not return single feature vectors for each observation.
Instead, it keeps a rolling window / buffer of the last N observations
so that it can reutrn a sequence of observations of length N.
This makes the training process simpler.
'''

SEQUENCE_LENGTH = 20
TRADING_COST_BPS = 1e-3
STEPS = 252

class TradingEnv(gym.Env):

    '''
    Sample usage: 
    - env = TradingEnv()
    - state = env.reset()
    - action = env.action_space.sample()
    - next_state, reward, done, _ = env.step(action)
    '''

    def __init__(self, steps=STEPS, sequence_length=SEQUENCE_LENGTH):
        super(TradingEnv, self).__init__()
        self.sequence_length = sequence_length
        self.steps = steps
        self.current_step = 0
        # IMPORTANT!! This file path is relative to where you run the script that imports this class
        self.data = get_data("GOOG")
        self.episode = -1
        # self.asset = 10_000

        self.actions = np.zeros(self.steps)
        self.rtn = np.ones(self.steps)
        self.mkt_rtn = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)

        # Action and observation space
        self.action_space = spaces.Discrete(3)  # 0, 1, 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(sequence_length, 10), dtype=np.float32)

        # Initialize state buffer
        self.state_buffer = np.zeros((sequence_length, 10))

    def reset(self, seed=None, options=None):

        '''
        You can access the state_buffer (sequence of observations) by env.reset()[0]
        @param: none
        @returns: state_buffer, {}
        '''

        self.current_step = 0
        # self.asset = 10_000
        self.actions = np.zeros(self.steps)
        self.rtn = np.ones(self.steps)
        self.mkt_rtn = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        # Reset the state buffer
        self.state_buffer = np.zeros((self.sequence_length, 10), dtype=np.float32)

        if self.episode != -1:
            # Each episode uses data from a different year
            # The first time we call reset is to initialize agent (episode = -1)
            # Then we call reset for episode 0, 1, 2, etc.
            start_idx = self.episode * 252
            end_idx = start_idx + 252
            self.df = self.data.iloc[start_idx:end_idx].copy()
        
        self.episode += 1
        
        return (self.state_buffer, {})

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        logging.info(f"{action} ({type(action)}) invalid" )

        # Update the state sequence with the latest observation
        obs = self.df.iloc[self.current_step]
        # Since obs is a series not a df we need to use labels not columns
        features = obs.drop(labels=["Return"])
        self.state_buffer = np.roll(self.state_buffer, shift=-1, axis=0)
        self.state_buffer[-1] = features

        self.mkt_rtn[self.current_step] = obs['Return']
        self.actions[self.current_step] = action

        # This represents the position at the beginning of the day (bod), 0, 1, or 2 (short, flat, long)
        bod_position = 0.0 if self.current_step == 0 else self.positions[self.current_step - 1]
        # This is an array that stores the position of the agent over time
        self.positions[self.current_step] = action - 1 # here -1 is short, 0 is flat, 1 is long

        self.trades[self.current_step] = self.positions[self.current_step] - bod_position # trade size, e.g. from long to short would be -1 - 1 = -2
        self.costs[self.current_step] = abs(self.trades[self.current_step]) * TRADING_COST_BPS
        
        # For now we don't use time cost to simplify the problem
        # Assume daily trading executed at the close
        self.rtn[self.current_step] = (bod_position * obs['Return']) - self.costs[self.current_step]
        reward = self.calculate_sortino_ratio()

        # self.asset = self.asset - self.costs[self.current_step] + self.trades[self.current_step] * obs["Adj Close"]

        # In OpenAI Gym v26 "done" is removed for "terminated" and "truncated"
        # Terminated: reached terminal state in MDP/ goal state
        # Truncated: end prematurely before a terminal state reached (e.g. timelimit, out of bounds)

        terminated = self.check_termination()
        truncated = self.check_truncation()

        self.current_step += 1

        return self.state_buffer, reward, terminated, truncated, {}
    
    # This function is the key reward function and requires careful data handling
    def calculate_sortino_ratio(self, risk_free_rate=0.04):
        
        # we will only use the last month's returns to calculate the sortino ratio
        rewards = self.rtn[:(self.current_step+1)]
        
        negative_returns = rewards[rewards < 0]

        assert len(rewards) > 0, f"Rewards list is empty at timestep {self.current_step} of episode {self.episode}"

        mean_return = np.mean(rewards)
        downside_deviation = np.std(negative_returns) + 1e-8 if len(negative_returns) > 0 else 1e-8

        try:
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
        except ZeroDivisionError:
            print(f"Standard deviation is {downside_deviation}, given list of returns: {self.rtn}, \
                at timestep {self.current_step} of episode {self.episode}")

        return sortino_ratio

    def check_termination(self):

        '''
        More conditions can be added here to terminate the episode early (e.g. negative asset)
        Each episode is a trading year with 252 trading days as 252 timesteps
        '''

        # Other examples may be the agent solved the environment...

        return True if self.current_step >= (self.steps - 1) else False
    
    def check_truncation(self):

        # In reality we would check like negative assets, etc.

        return False

print("Environment imported successfully")