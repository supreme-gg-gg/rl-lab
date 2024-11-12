import numpy as np
from config import *

class ReplayBuffer():
    def __init__(self, env, buffer_capacity=BUFFER_CAPACITY,
                 batch_size = BATCH_SIZE, min_size_buffer=MIN_SIZE_BUFFER) -> None:
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
    
        # The buffer can also be made using a deque and named tuple, but functions in identical ways
        self.states = np.zeros((self.buffer_capacity, env.observation_space.shape[0]))
        self.actions = np.zeros((self.buffer_capacity, env.action_space.shape[0]))
        self.rewards = np.zeros((self.buffer_capacity))
        self.next_states = np.zeros((self.buffer_capacity, env.observation_space.shape[0]))
        self.dones = np.zeros((self.buffer_capacity), dtype=bool)


    def __len__(self):
        return self.buffer_counter
    

    def push(self, state, action, reward, next_state, done):

        # set index to zero if counter = buffer_cap and start again...
        index = self.buffer_counter % self.buffer_capacity
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.buffer_counter += 1

    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer

    def update_n_games(self):
        self.n_games += 1

    def sample(self):
        # if cnt < cap we don't want to take zero records
        # if cnt higher don't access using cnt because older records are deleted

        buffer_range = min(self.buffer_counter, self.buffer_capacity)
        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)

        state = self.states[batch_index]
        action = self.actions[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]

        return state, action, reward, next_state, done
    
    # We will not save and load replay buffer but doing it might save time...
    # It allows you to stop and restart training (e.g. Colab)