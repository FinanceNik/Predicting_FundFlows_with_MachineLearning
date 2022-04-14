import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout

data = pd.read_csv('data/test_data.csv')
print(data['Monthly Gross Return \n2000-06 \nBase \nCurrency'])


def neural_net():
    data_file = 'data/MutualFund prices - A-E.csv'
    data = pd.read_csv(data_file)
    data = data.loc[(data['fund_symbol'] == 'AAAAX')]
    # print(data.columns[:])
    print(data[-2:-1])
    predictor = 'nav_per_share'

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[predictor].values.reshape(-1, 1))

    prediction_period = 60

    x_train = []
    y_train = []

    for x in range(prediction_period, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_period:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True,))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=40, batch_size=32)

    ''' Testing the Model'''

    test_data = data[:]

    actual_prices = test_data[predictor].values

    total_dataset = pd.concat((data[predictor], test_data[predictor]))

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_period:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(prediction_period, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_period:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.plot(actual_prices, color='black', label=f'Actual predictor')
    plt.plot(predicted_prices, color='green', label=f'Predicted predictor')
    plt.legend()
    plt.show()

    ''' Predicting the Next Period'''

    real_data = [model_inputs[len(model_inputs) + 0 - prediction_period:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(prediction)

#
# neural_net()


