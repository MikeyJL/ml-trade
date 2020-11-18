import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools
from model import nn_predict


class TradingEnv(gym.Env):
  def __init__(self, train_data, init_invest=500):
    # data
    self.stock_price_history = train_data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.init_invest = init_invest
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    # action space
    self.action_space = spaces.Discrete(3**self.n_stock)

    # observation space: give estimates in order to sample and build scaler
    stock_max_price = self.stock_price_history.max(axis=0)
    stock_range = [[0.00001, init_invest * 2 / mx] for mx in stock_max_price]
    price_range = [[0.00001, mx] for mx in stock_max_price]
    cash_in_hand_range = [[1, init_invest * 2]]
    self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

    # seed and start
    self._seed()
    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def _reset(self):
    self.cur_step = 0
    self.stock_owned = [0] * self.n_stock
    self.stock_price = self.stock_price_history[:, self.cur_step]
    self.cash_in_hand = [self.init_invest]
    return self._get_obs()


  def _step(self, action):
    assert self.action_space.contains(action)
    prev_val = self._get_val()
    self.cur_step += 1
    self.stock_price = self.stock_price_history[:, self.cur_step]
    self._trade(action)
    cur_val = self._get_val()
    reward = cur_val - prev_val
    done = self.cur_step == self.n_step - 1
    info = {'cur_val': cur_val}
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = []
    obs.append(self.stock_owned)
    obs.append(nn_predict(self.stock_price_history))
    obs.append(self.cash_in_hand)
    print(obs)
    return obs


  def _get_val(self):
    return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand


  def _trade(self, action):
    # all combo to sell(0), hold(1), or buy(2) stocks
    action_combo = map(list, itertools.product([0, 1, 2], repeat=self.n_stock))
    action_vec = action_combo[action]

    # one pass to get sell/buy index
    sell_index = []
    buy_index = []
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # two passes: sell first, then buy; might be naive in real-world settings
    if sell_index:
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    if buy_index:
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1 # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False