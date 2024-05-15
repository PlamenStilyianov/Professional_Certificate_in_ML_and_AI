

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

df_stocks = pd.read_csv('data/jpm_bars_2024-03-06.csv')

# Reset the index
df_stocks['timestamp'] = df_stocks.timestamp.apply(lambda x: x.split(' ')[0])
df_stocks.rename(columns={'timestamp': 'date'}, inplace=True)
df_stocks.drop(['symbol'], axis=1, inplace=True)
df_stocks.set_index('date', inplace=True)
df_stocks.head()

df = df_stocks[['vwap', 'high', 'low', 'close']]


# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(df) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
HLC_avg = df[['high', 'low', 'close']].mean(axis=1)
VWAP_avg = df[['vwap']]
close_val = df[['close']]
plt.figure(figsize=(15, 7))
plt.plot(obs, HLC_avg, 'g', label='HLC avg')
plt.plot(obs, VWAP_avg, 'b', label='VWAP price')
plt.plot(obs, close_val, 'r', label='Close price')
plt.title('Historical Closing Price of JP Morgan Chase (JPM) stock')
plt.legend(loc='lower right')
plt.show()

# RESHAPING OF THE DATASET UTIL
def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset) - step_size - 1):
        a = dataset[i:(i + step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)

# PREPARATION OF TIME SERIES DATASET
VWAP_avg = np.reshape(VWAP_avg.values, (len(VWAP_avg), 1))  # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
VWAP_avg = scaler.fit_transform(VWAP_avg)

# TRAIN-TEST SPLIT
train_VWAP = int(len(VWAP_avg) * 0.75)
test_VWAP = len(VWAP_avg) - train_VWAP
train_VWAP, test_VWAP = VWAP_avg[0:train_VWAP, :], VWAP_avg[train_VWAP:len(VWAP_avg), :]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train_VWAP, 1)
testX, testY = new_dataset(test_VWAP, 1)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))
# model = Sequential()
# model.add(LSTM(32, input_shape=(1, step_size), return_sequences=True))
# model.add(LSTM(16))
# model.add(Dropout(0.25))
# model.add(Dense(1))
# model.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])  # Try SGD, adam, adagrad and compare!!!
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2, validation_data=(testX, testY))
# model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'] )  # Try SGD, adam, adagrad and compare!!!
# fine_tuned_history = model.fit(trainX, trainY, epochs=50, batch_size=1, shuffle=True, verbose=2, validation_data=(testX, testY))
# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train RMSE: %.2f' % (trainScore))

# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test RMSE: %.2f' % (testScore))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(VWAP_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict) + step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(VWAP_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (step_size * 2) + 1:len(VWAP_avg) - 1, :] = testPredict

# DE-NORMALIZING MAIN DATASET 
VWAP_avg = scaler.inverse_transform(VWAP_avg)

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.figure(figsize=(15, 7))
plt.plot(VWAP_avg, 'g', label='original dataset')
plt.plot(trainPredictPlot, 'r', label='training set')
plt.plot(testPredictPlot, 'b', label='predicted stock price/test set')
plt.legend(loc='lower right')
plt.xlabel('Time in Days')
plt.ylabel('VWAP Value of JPM Stocks')
plt.show()

# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val / last_val
next_val = model.predict(np.reshape(last_val_scaled, (1, 1, 1)))
print("Last Day Value:", np.ndarray.item(last_val))
print("Next Day Value:", np.ndarray.item(last_val * next_val))


#%%

#%%
