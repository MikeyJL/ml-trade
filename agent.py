from collections import deque
import random
import numpy as np
from model import dqn
import matplotlib.pyplot as plt


class DQNAgent(object):
  """ A simple Deep Q agent """
  def __init__(self, state_size, action_size, price_data):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gamma = 0.8
    self.epsilon = 1
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.99
    self.model = dqn(state_size, action_size)
    self.action_history = []
    self.bal_history = []

    price_data = np.array(price_data).flatten()
    self.price_data = price_data


  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    self.action_history.append(action)


  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])


  def replay(self, batch_size=32):
    minibatch = random.sample(self.memory, batch_size)
    states = np.array([tup[0] for tup in minibatch])
    actions = np.array([tup[1] for tup in minibatch])
    rewards = np.array([tup[2] for tup in minibatch])
    next_states = np.array([tup[3] for tup in minibatch])
    done = np.array([tup[4] for tup in minibatch])

    # Q(s', a)
    target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
    target[done] = rewards[done]
    
    # Q(s, a)
    target_f = self.model.predict(states)
  
    for i in range(batch_size):
      target_f[i, actions[i]] = target[0][i]
    self.model.fit(states, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)


  def _plot_graph(self, graph_name):
    x1 = []
    x2 = []
    for i in range(len(self.action_history)):
      x1.append(i + 1)
    for i in range(len(self.price_data)):
      x2.append(i + 1)
    fig, axs = plt.subplots(3)
    fig.suptitle('Price, Balance and Actions')
    axs[0].plot(x2, self.price_data)
    axs[1].plot(x1, self.action_history)
    axs[2].plot(x1, self.bal_history)
    plt.savefig('figures/{}.png'.format(graph_name))