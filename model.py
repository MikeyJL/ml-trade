import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def dqn(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
  model = Sequential()
  model.add(LSTM(n_neuron_per_layer, input_shape=n_obs, return_sequences=True, activation=activation))
  for _ in range(n_hidden_layer):
    model.add(LSTM(n_neuron_per_layer, return_sequences=False, activation=activation))
  model.add(Dense(n_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model


def nn_predict(dataset):
  training_sample_80 = math.ceil(len(dataset) * .8)
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(dataset)
  train_data = scaled_data[0:training_sample_80, :]
  currency_samples = []
  currency_labels = []

  for i in range(30, len(train_data)):
    currency_samples.append(train_data[i-30:i, 0])
    currency_labels.append(train_data[i, 0])
  currency_samples, currency_labels = np.array(currency_samples), np.array(currency_labels)
  currency_samples = np.reshape(currency_samples, (currency_samples.shape[0], currency_samples.shape[1], 1))

  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(currency_samples.shape[1], 1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
  model.fit(currency_samples, currency_labels, batch_size=1, epochs=1, verbose=2)
  test_data = scaled_data[training_sample_80 - 30:, :]
  test_samples = []
  test_labels = dataset[training_sample_80:, :]

  for i in range(30, len(test_data)):
    test_samples.append(test_data[i-30:i, 0])
  test_samples = np.array(test_samples)
  test_samples = np.reshape(test_samples, (test_samples.shape[0], test_samples.shape[1], 1))
  predictions = model.predict(test_samples)
  predictions = scaler.inverse_transform(predictions)
  rmse = np.sqrt(np.mean(predictions - test_labels)**2)

  predict_next_df = dataset
  last_30_days = predict_next_df[-30:]
  last_30_days_scaled = scaler.transform(last_30_days)
  get_predict_samples = []
  get_predict_samples.append(last_30_days_scaled)
  get_predict_samples = np.array(get_predict_samples)
  get_predict_samples = np.reshape(get_predict_samples, (get_predict_samples.shape[0], get_predict_samples.shape[1], 1))
  predicted_price = model.predict(get_predict_samples)
  predicted_price = scaler.inverse_transform(predicted_price)

  return predicted_price
