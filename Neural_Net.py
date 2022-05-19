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
import pickle


def neural_network_classification():

    df = dsc.ml_algo_selection('classifier')

    # df = df.sample(100_000)

    X = df.drop(['fund_flow'], axis=1).values
    y = df['fund_flow'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # (2, activation=[tf.nn.softmax])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(x=X_train, y=Y_train, epochs=15, validation_data=(X_test, Y_test), verbose=1)

    val_loss, val_acc = model.evaluate(X_test, Y_test)
    print(f'loss: {val_loss}, acc: {val_acc}')


neural_network_classification()


