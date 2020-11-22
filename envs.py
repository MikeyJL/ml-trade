import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools
from model import nn_predict
from agent import DQNAgent


class TradingEnv(gym.Env):
  def __init__(self, train_data, mode):
    self.stock_price_history = train_data
    self.max_step, self.n_stock = self.stock_price_history.shape
    self.mode = mode
    self.stock_owned = None
    self.stock_price = None
    self.action_space = spaces.Discrete(3)
    self.observation_space = np.array(self._reset()).shape


  def _reset(self):
    self.cur_step = 0
    self.stock_owned = 0
    if self.mode == 'train':
      self.stock_price = self.stock_price_history[self.cur_step]
      self.cash_start = 100
      self.cash_bal = 100
    return self._get_obs()


  def _step(self, action):
    assert self.action_space.contains(action)
    self.cur_step += 1
    self._trade(action)
    if self.mode == 'train':
      prev_stock_price = self.stock_price
      self.stock_price = self.stock_price_history[self.cur_step]
      reward = self.stock_owned * (self.stock_price - prev_stock_price)
      self.cash_bal += reward
    self.last_reward = reward
    done = self.cur_step == self.max_step - 1
    return self._get_obs(), reward, done


  def _get_obs(self):
    obs = []
    obs.append(self.stock_price)
    return obs


  def _trade(self, action):
    if self.mode == 'train':
      if action == 0:
        self.cash_bal += 5 * self.stock_price
        self.stock_owned = 0
        self.last_trade = 'SELL'
      elif action == 2:
        self.stock_owned += 5
        self.cash_bal -= 5 * self.stock_price
        self.last_trade = 'OPEN'
      else:
        self.last_trade = 'HOLD'