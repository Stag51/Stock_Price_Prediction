#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#Importing the training set

ds_train = pd.read_csv('NSE-TATAGLOBAL.csv')
training_set = ds_train.iloc[:, 1:2].values

#print(ds_train.head())

#Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []

for i in range(60, 2035):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] , 1))

#Initialising the RNN

regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularization

regressor.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding Second LSTM Layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding Fourth Layer of LSTM and Dropout REgularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer

regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training Set
#regressor.fit(x_train, y_train, epochs = 100, batch_size= 32)

#Getting the real stock price
dataset_test = pd.read_csv('tatatest.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price

dataset_total = pd.concat((ds_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 76):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predict_stock_price = regressor.predict(x_test)
predict_stock_price = sc.inverse_transform(predict_stock_price)


#Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real TATA Stock Price')
plt.plot(predict_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title("Stock Price prediction by Saad Shabbir")
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()