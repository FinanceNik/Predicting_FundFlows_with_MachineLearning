from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import DataSet_Cleaner as dsc
import tensorflow as tf
import numpy as np
import pandas as pd


def neural_network_extended_classification():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This is the neural network module used for the extended classification.

    """
    df = dsc.ml_algo_selection('extended_classifier')  # Calling the dataset for the extended classification.
    fund_flow_cats = df.filter(regex='fund_flow')
    fund_flow_cats = pd.get_dummies(fund_flow_cats, columns=['fund_flow'])
    df = df.drop(['fund_flow'], axis=1)

    X = df
    y = fund_flow_cats

    col_len = len(list(X.columns[:]))

    # Splitting the test and the training dataset in order to eliminate bias.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    epochs = 1500  # The epochs can be set arbitrarily high as the early stopping mechanism deciding on when to stop.

    model = Sequential()
    model.add(tf.keras.layers.Flatten())  # flattening the input gives slightly better results.
    model.add(Dense(col_len, input_shape=(X.shape[1],), kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(col_len, activation='relu'))
    model.add(Dense(col_len, activation='relu'))
    model.add(Dense(round(col_len/2, 0), activation='relu'))
    model.add(Dense(20, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       patience=4,
                       restore_best_weights=True)

    # Callbacks allow for the early stopping of a model if over-fitting is detected.
    model.fit(x=X_train, y=Y_train, callbacks=[es], batch_size=10,
              epochs=epochs, validation_data=(X_test, Y_test), verbose=1)

    val_loss, val_acc = model.evaluate(X_test, Y_test)
    print(f'loss: {val_loss}, acc: {val_acc}')

    model.predict(X_test)
    np.round(model.predict(X_test), 0)

    preds = np.round(model.predict(X_test), 0)

    report = classification_report(Y_test, preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('ext_report_nn.csv')


neural_network_extended_classification()




