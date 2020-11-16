import gym
import math
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import argparse
import time

from envs import TradingEnv
from agent import DQNAgent
from utility import get_data, maybe_make_dir

# PARAMS

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
alpha = 0.001
num_episodes = 1000

# RUN

## Makes cmd for script
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--episode', type=int, default=2000,
                    help='number of episode to run')
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='batch size for experience replay')
parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                    help='initial investment amount')
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
args = parser.parse_args()

# Creates new files to store data
maybe_make_dir('weights')
maybe_make_dir('portfolio_val')

# Gets time
timestamp = time.strftime('%Y-%m-%d')

# Set datas
data = get_data('Close', 'USDGBP=X', '2012-01-01' timestamp)
train_samples = data[:, data.shape[0]:]
test_samples = data[:, data.shape[0]:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TradeEnvManager(device)
state_size = env.observation_space.shape
action_size = env.action_space.n
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, env.num_actions_available(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN(env.get_screen_height(), env.get_screen_width()).to(device)
target_net = DQN(env.get_screen_height(), env.get_screen_width()).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), alpha=alpha)

# TRAIN

episode_durations = []

for episode in range(num_episodes):
  env.reset()
  state = env.get_state()

  for timestep in count():
    action = agent.select_action(state, policy_net)
    reward = env.take_action(action)
    next_state = env.get_state()
    memory.push(Experience(state, action, next_state, reward))
    state = next_state

    if memory.can_provide_sample(batch_size):
      experiences = memory.sample(batch_size)
      states, actions, rewards, next_states = extract_tensors(experiences)

      current_q_values = QValues.get_current(policy_net, states, actions)
      next_q_values = QValues.get_next(target_net, next_states)
      target_q_values = (next_q_values * gamma) + rewards

      loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
    if env.done:
      episode_durations.append(timestep)
      plot(episode_durations, 100)
      break
  
  if episode % target_update == 0:
    target_net.load_state_dict(policy_net.state_dict())

env.close()
