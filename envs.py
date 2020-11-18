import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools
from model import nn_predict
from ig_api import LiveData


class TradingEnv(gym.Env):
  def __init__(self, train_data, steps):
    # data
    self.stock_price_history = train_data
    self.max_step = steps
    self.n_stock = self.stock_price_history.shape[1]

    # instance attributes
    self.init_invest = LiveData()._get_account()['accounts'][1]['balance']['balance']
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    # action space
    self.action_space = spaces.Discrete(3**self.n_stock)

    # observation space
    self.observation_space = np.array(self._reset()).shape

    # seed and start
    self._seed()
    self._reset()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def _reset(self):
    self.cur_step = 0
    self.stock_owned = 0 * self.n_stock
    self.stock_price = LiveData()._get_current_price()['snapshot']['bid']
    self.cash_in_hand = LiveData()._get_account_val()
    return self._get_obs()


  def _step(self, action):
    assert self.action_space.contains(action)
    prev_val = self.cash_in_hand
    self.cur_step += 1
    self.stock_price = LiveData()._get_current_price()['snapshot']['bid']
    self._trade(action)
    cur_val = LiveData()._get_account_val()
    reward = cur_val - prev_val
    done = self.cur_step == self.max_step - 1
    info = {'cur_val': cur_val}
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = []
    obs.append([self.stock_owned])
    obs.append([self.stock_price])
    obs.append([self.cash_in_hand])
    return obs


  def _trade(self, action):
    if action == 0:
      LiveData()._open_position('SELL')
      print('BUY')
    elif action == 2:
      LiveData()._open_position('BUY')
      print('SELL')
    else:
      print('HOLD')