#!/usr/bin/python

#******************************
#******** Enviroment **********
#******************************

ENV_NAME = 'BipedalWalkerHardcore-v3'
# ENV_NAME = 'LunarLanderContinuous-v2'
# ENV_NAME = 'Pendulum-v1'

# NUM_ENVS = 4
# TRAINING_STEPS_PER_ENV_STEP = 1

PATH_SAVE = "../models/bipedal"
PATH_LOAD = "../models/bipedal"
#PATH_LOAD = "../model/save_agent_202102130731"

#******************************
#****** Replay Buffer *********
#******************************

BATCH_SIZE = 128
MIN_SIZE_BUFFER = 100 # Minimum size of the buffer to start learning, until then random actions
BUFFER_CAPACITY = 1_000_000

#******************************
#******** Networks ************
#******************************

ACTOR_HIDDEN_0 = 256
ACTOR_HIDDEN_1 = 128
#INIT_MINVAL = -0.05
#INIT_MAXVAL = 0.05

CRITIC_HIDDEN_0 = 256
CRITIC_HIDDEN_1 = 128

#******************************
#********** Agent *************
#******************************

GAMMA = 0.99
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
# ACTOR_LR = 0.01
# CRITIC_LR = 0.005

TAU = 1e-3 # For soft update the target network

# Parameters for Ornstein–Uhlenbeck process
THETA=0.15
DT=1e-1

#******************************
#********** Main **************
#******************************

MAX_GAMES = 2000
SAVE_FREQUENCY = 200
ENABLE_WANDB = True
MAX_TIMESTEPS = 700
