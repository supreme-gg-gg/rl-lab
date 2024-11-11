import random, math, json
from collections import deque, namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gdqn.utils import plot_durations

import logging
logging.basicConfig(filename="training.log", level=logging.INFO)

with open("config.json", "r") as f:
    config = json.load(f)

MEMORY_SIZE = config["MEMORY_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
GAMMA = config["GAMMA"]
EPS = config["EPS"]
EPS_MIN = config["EPS_MIN"]
EPS_DECAY = config["EPS_DECAY"]
TAU = config["TAU"]
LR = config["LR"]
NUM_STEPS_TO_UPDATE = config["NUM_STEPS_TO_UPDATE"]

HIDDEN_SIZE_GRU = config["HIDDEN_SIZE_GRU"]
NUM_LAYERS_GRU = config["NUM_LAYERS_GRU"]
DROPOUT = config["DROPOUT"]

# ======================Replay Buffer=======================

Transition = namedtuple("Transition", ("sequence", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# For now we assume transitions are totally deterministic

'''
    - Num_layers is the number of GRU units stacked on top of each other
    - Hidden_size is the number of features in the hidden state h
    - Output_size is the number of features in the output (e.g. number of actions)
'''

class GDQN(nn.Module):

    '''
    Gated Deep Q Network
    GRU -> Dropout -> GRU -> Dropout -> Linear
    Output tensor: [batch_size, output_size]
    '''

    def __init__(self, input_size, output_size):
        super(GDQN, self).__init__()

        # PyTorch automatically initializes hidden state to 0 if not provided
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=HIDDEN_SIZE_GRU, num_layers=NUM_LAYERS_GRU, batch_first=True)
        self.dropout1 = nn.Dropout(p=DROPOUT)
        self.gru2 = nn.GRU(input_size=HIDDEN_SIZE_GRU, hidden_size=HIDDEN_SIZE_GRU, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE_GRU, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x += 1e-8  # stabilize training

        x, _ = self.gru1(x)
        x = self.dropout1(x)
        
        x, _ = self.gru2(x)
        x = self.dropout2(x)

        # We only need the last timestep output for the fully-connected layer
        x = self.fc(x[:, -1, :])

        return x
    
class DQNAgent():

    def __init__(self, env, device):
        n_observations, n_features = env.reset()[0].shape # returns a sequence of observations in shape (sequence_length, n_features)
        n_actions = env.action_space.n

        policy_net = GDQN(n_features, n_actions).to(device)
        target_net = GDQN(n_features, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        self.env = env
        self.device = device
        self.n_actions = n_actions
        self.n_observations = n_observations # this is equivalent to sequence_length, which is defined in the env
        self.n_features = n_features
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        self.num_steps_to_update = NUM_STEPS_TO_UPDATE
    
    def select_action(self, state):

        eps_threshold = EPS_MIN + (EPS - EPS_MIN) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                x = self.policy_net(state)
                if x.max(1).indices.view(1, 1).item() == -1:
                    print("Action is -1")
                    print("State: ", state)
                    print(x)
                return x.max(1).indices.view(1, 1)
        else:
            action = self.env.action_space.sample()
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):

        # In the modified implementation, each batch contains BATCH_SIZE of sequences
        # In other words, each state entry in memory buffer is a sequence of observations

        if len(self.memory) < BATCH_SIZE or self.steps_done % self.num_steps_to_update != 0:
            return
        
        # Sample a random batch of transitions and unpack them into their own tensors
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Remove the leading dimension from each tensor in batch.sequence
        # since each tensor in batch.sequence is [1, 20, 10] where 1 is BATCH_SIZE by default
        # Concatenate sequences to form batch of sequences for training
        state_batch = torch.stack([s.squeeze(0) for s in batch.sequence]) # (BATCH_SIZE, SEQ_LENGTH, input_size)
        action_batch = torch.stack([s.squeeze(0) for s in batch.action]) # (BATCH_SIZE, 1) scalar action
        reward_batch = torch.stack([s.squeeze(0) for s in batch.reward]) # (BATCH_SIZE, 1) scalar reward

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s.squeeze(0) for s in batch.next_state if s is not None]) # (BATCH_SIZE, SEQ_LENGTH, input_size)

        # Compute Q(s_t, a) - the model computes Q(s_t), then select the action(s) that were actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s') for all next states, expected values of actions for 
        # non_final_next_states are based on the older target_net, select best reward
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # just select max here since we assume the agent acts ideally afterwards
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values # max along feature dimension
            # The resultant next_state_values tensor is of shape [BATCH_SIZE] (1D tensor)

        # Compute the expected Q values: y = r + gamma * max_a(Q(s', a))
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Unlike the TF example we are using Huber Loss here (look it up)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train_agent(self, num_episodes):

        '''
        Args:
        - num_episodes (int): number of episodes to train the agent

        Returns:
        - episode_durations (list): list containing the duration of each episode
        '''

        episode_durations = []

        for i in range(num_episodes):

            state = self.env.reset()[0]
            # unsqueeze adds a batch dimension to the tensor
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model() # on the policy network

                # soft update target network weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in target_net_state_dict:
                    target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    # duration is how long it took to run that episode
                    episode_durations.append(t + 1)
                    plot_durations(episode_durations)
                    break

                logging.info(f"Episode {i} completed in {t + 1} steps")
        
        # IMPORTANT: the path of model is relative to where the script is run
        # for now it is ONLY from the gdqn-model.ipynb notebook
        # but in the future we might need to make it more flexible!!
        torch.save(self.policy_net.state_dict(), "../../models/gdqn_trained.pth")
        print("Model saved as gdqn_trained.pth")

        return episode_durations

print("Agent Module Loaded")