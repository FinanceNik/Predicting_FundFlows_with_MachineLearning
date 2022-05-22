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


def neural_network_classification():
    df = dsc.ml_algo_selection('classifier')
    # df = df.sample(50_000)
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

    val_loss, val_acc = model.evaluate(X_test, Y_test)
    print(f'loss: {val_loss}, acc: {val_acc}')

    model.predict(X_test)
    np.round(model.predict(X_test), 0)

    preds = np.round(model.predict(X_test), 0)

    Statistics.confusion_matrix(confusion_matrix(Y_test, preds), 'Neural Network Classification')

    report = classification_report(Y_test, preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    # df.to_csv('report_nn.csv')

    try:
        feature_imp = model.feature_importances_
        feature_names = list(df.drop(drops, axis=1).columns[:])
        Statistics.feature_importance(feature_names, feature_imp, 'Neural Net for Classification')
    except:
        pass


# neural_network_classification()


