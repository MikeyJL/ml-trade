import pickle
import time
import numpy as np
import argparse
import re
import os

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, maybe_make_dir, get_scaler

if __name__ == '__main__':
  os.system('clear')
  timestamp = time.strftime('%Y%m%d')

  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episodes', type=int, default=1,
                      help='number of episodes to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test" by entering target weights file')
  parser.add_argument('-d', '--delay', type=int, default=60,
                      help='delay between steps')
  parser.add_argument('-w', '--weights', type=str, default=None,
                      help='delay between steps')
  parser.add_argument('-g', '--graph', type=str, default=None,
                      help='delay between steps')
  args = parser.parse_args()

  maybe_make_dir('weights')

  data = get_data()
  train_data = data[:data.shape[0]:]
  test_data = data[:data.shape[0]:]

  env = TradingEnv(train_data, args.mode)
  state_size = env.observation_space
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size, train_data)

  if args.mode != 'train':
    env = TradingEnv(test_data, args.mode)
    agent.load(args.mode)

  for e in range(args.episodes):
    agent.epsilon = 1
    state = env._reset()
    scaler = get_scaler(env)
    state = scaler.transform(state)
    for s in range(env.max_step):
      state = np.reshape(state, (1, state.shape[0], 1))
      action = agent.act(state)
      next_state, reward, done = env._step(action)
      next_state = scaler.transform(next_state)
      if args.mode == 'train':
        agent.bal_history.append(env.cash_bal[0])
        state = np.reshape(state, (state.shape[1], 1))
        agent.remember(state, action, reward, next_state, done)
      state = next_state
      print('Step: {}/{}, Trade: {}, Stocks: {}, Bal: {} Reward: {} --- Epsilon: {}'
            .format(s + 1, env.max_step, env.last_trade,
                    np.around(env.stock_owned, decimals=1),
                    np.around(env.cash_bal[0], decimals=2),
                    np.around(env.rewarded[0], decimals=4),
                    np.around(agent.epsilon, decimals=2)))
      if done:
        print("Episode: {}/{}".format(e + 1, args.episodes))
        break
      if args.mode == 'train' and len(agent.memory) > args.batch_size:
        agent.replay(args.batch_size)
      if args.mode != 'train':
        time.sleep(args.delay)
    if args.mode == 'train' and args.weights != None:
      maybe_make_dir('weights/{}'.format(args.weights))
      agent.save('weights/{}/{}_({}).h5'.format(args.weights, timestamp, int(env.cash_bal[0])))

if args.graph != None:
  agent._plot_graph(args.graph)