import pickle
import time
import numpy as np
import argparse
import re
from sklearn.preprocessing import MinMaxScaler

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, maybe_make_dir

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episodes', type=int, default=1,
                      help='number of episodes to run')
  parser.add_argument('-s', '--steps', type=int, default=3,
                      help='number of steps to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  parser.add_argument('-w', '--weights', type=str,
                      help='a trained model weights')
  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')

  timestamp = time.strftime('%Y%m%d%H%M')

  data = get_data()
  train_data = data[:data.shape[0]:]
  test_data = data[:data.shape[0]:]

  env = TradingEnv(train_data, args.steps)
  state_size = env.observation_space
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  scaler = MinMaxScaler(feature_range=(0, 1))
  portfolio_value = []

  if args.mode == 'test':
    # remake the env with test data
    env = TradingEnv(test_data, args.initial_invest)
    # load trained weights
    agent.load(args.weights)
    # when test, the timestamp is same as time when weights was trained
    timestamp = re.findall(r'\d{12}', args.weights)[0]

  for e in range(args.episodes):
    state = env._reset()
    state = scaler.fit_transform(state)
    for i in range(env.max_step):
      action = agent.act(state)
      next_state, reward, done, info = env._step(action)
      next_state = scaler.fit_transform(next_state)
      if args.mode == 'train':
        agent.remember(state, action, reward, next_state, done)
      state = next_state
      if done:
        print("episode: {}/{}, episode end value: {}".format(
          e + 1, args.episodes, info['cur_val']))
        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        break
      if args.mode == 'train' and len(agent.memory) > args.batch_size:
        agent.replay(args.batch_size)
      time.sleep(10)
    if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
      agent.save('weights/{}-dqn.h5'.format(timestamp))

  # save portfolio value history to disk
  with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    pickle.dump(portfolio_value, fp)