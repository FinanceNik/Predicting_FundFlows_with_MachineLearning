from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import DataSet_Cleaner as dsc
import numpy as np
import Statistics
import pickle
from sklearn import metrics
import os


def neural_network(min, max, n):
    df = dsc.ml_algo_selection('regression')
    df = df[(df["fund_flow"] < max)]
    df = df[(df["fund_flow"] > min)]
    df = df.sample(n)

    def scaling(x):
        x = (2 * (x - min) / (max - min)) - 1
        return x

    df['fund_flow'] = df['fund_flow'].apply(scaling)

    col_len = len(df.drop(['fund_flow'], axis=1).columns[:])

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    epochs = 80

    model = Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(col_len, input_shape=(X.shape[1],), activation='relu'))  # 128 is best atm. && include input_shape argument
    model.add(Dense(col_len, activation='relu'))
    model.add(Dense(col_len, activation='relu'))
    model.add(Dense(col_len, activation='relu'))
    model.add(Dense(col_len, activation='relu'))
    model.add(Dense(round(col_len/2, 0), activation='relu'))  # create another /2 layer ? OR remove half-layer?
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',  # don't minimize the accuracy!
                       patience=8,
                       restore_best_weights=True)

    hist = model.fit(x=X_train, y=Y_train, callbacks=[es], batch_size=10,
                     epochs=epochs, validation_data=(X_test, Y_test), verbose=1)  # include batch size

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # Statistics.loss_visualizer(train_loss, val_loss, len(hist.history['loss']))
    # Statistics.accuracy_visualizer(train_acc, val_acc, len(hist.history['loss']))

    # val_loss, val_acc = model.evaluate(X_test, Y_test)
    # print(f'loss: {val_loss}, acc: {val_acc}')

    model.predict(X_test)
    np.round(model.predict(X_test), 0)

    Y_pred = np.round(model.predict(X_test), 0)

    mean_absolute_error = metrics.mean_absolute_error(Y_test, Y_pred)
    mean_squared_error = metrics.mean_squared_error(Y_test, Y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    r2_score = metrics.r2_score(Y_test, Y_pred)

    try:
        explained_variance_score = metrics.explained_variance_score(Y_test, Y_pred)
    except:
        explained_variance_score = 'none'

    try:
        d2_absolute_error_score = metrics.d2_absolute_error_score(Y_test, Y_pred)
    except:
        d2_absolute_error_score = 'none'

    file_name = 'metrics/regression_neuralNetwork.csv'
    cmd_header = f'echo echo "mean_absolut_error,mean_squared_error,root_mean_squared_error,r2_score,explained_variance_score,d2_absolute_error_score" >> {file_name}'
    cmd_data = f'echo "{mean_absolute_error},{mean_squared_error},{root_mean_squared_error},{r2_score},{explained_variance_score},{d2_absolute_error_score}" >> {file_name}'
    os.system(cmd_header)
    os.system(cmd_data)




