from sklearn.model_selection import train_test_split  # splitting the dataset into training and test sets
from sklearn.preprocessing import MinMaxScaler  # scaling the data according to best practice
from tensorflow.keras import Sequential  # the model used
from tensorflow.keras.layers import Dense, Dropout  # the nodes used in this model
import tensorflow as tf  # for calling the flattening layer at the beginning of the model
from tensorflow.keras.callbacks import EarlyStopping  # stopping the model's training early if over-fitted
import DataSet_Cleaner as dsc  # retrieving the correct dataset
import numpy as np  # for rounding values
import Statistics  # calling the visualization functions
from sklearn import metrics  # evaluation metrics
import os  # writing the evaluation metrics to a .csv file


def neural_network(min, max, n):
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This is the neural network used for the prediction of the actual values of the future fund
    flows. The [min, max, n] arguments stand for:
                                                  - minimum fund flow to be included
                                                  - maximum fund flow to be included
                                                  - number of the sample size
    These limitations are necessary because the fund flow variance is too high and the models give horrible
    predictions. Hence, in order to homogenize the data a bit, the outliers are excluded.

    """
    df = dsc.ml_algo_selection('regression')  # retrieving the correct dataset for the regression purpose
    df = df[(df["fund_flow"] < max)]  # filtering the highest outliers
    df = df[(df["fund_flow"] > min)]  # filtering the lowest outliers
    df = df.sample(n)  # take number of random samples

    def scaling(x):
        x = (2 * (x - min) / (max - min)) - 1
        return x

    df['fund_flow'] = df['fund_flow'].apply(scaling)  # applying the scaling function to the fund flows

    col_len = len(df.drop(['fund_flow'], axis=1).columns[:])

    predictor = 'fund_flow'  # predicting fund flows
    drops = [predictor]  # dropping fund flows from the predictor set

    X = df.drop(drops, axis=1).values  # predictors
    y = df[predictor].values  # predicted

    # Splitting the dataset into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=101)

    # Scaling the data according to the best practice
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    epochs = 800  # stopped early by the early stopping function

    # Creating the model
    model = Sequential()
    model.add(tf.keras.layers.Flatten())  # flatten the input according to best practice
    model.add(Dense(col_len, input_shape=(X.shape[1],), activation='relu'))  # Input dense layer
    model.add(Dense(col_len, activation='relu'))  # hidden layer with n_nodes == num of predicting features
    model.add(Dense(col_len, activation='relu'))  # hidden layer with n_nodes == num of predicting features
    model.add(Dense(col_len, activation='relu'))  # hidden layer with n_nodes == num of predicting features
    model.add(Dense(col_len, activation='relu'))  # hidden layer with n_nodes == num of predicting features
    model.add(Dense(round(col_len/2, 0), activation='relu'))  # hidden layer with n_nodes == (num of pred. features)/ 2
    model.add(Dense(1, activation='sigmoid'))  # best practice activation function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # defining the optimizer

    # Calling and defining the early stopping algorithm
    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       patience=8,
                       restore_best_weights=True)

    # Fitting the data to the model and training it
    hist = model.fit(x=X_train, y=Y_train, callbacks=[es], batch_size=10,
                     epochs=epochs, validation_data=(X_test, Y_test), verbose=1)

    # Different metrics by which neural networks are to be evaluated graphically
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # Calling the visualizing functions that show graphically whether the neural network is fitted well or not
    Statistics.loss_visualizer(train_loss, val_loss, len(hist.history['loss']))
    Statistics.accuracy_visualizer(train_acc, val_acc, len(hist.history['loss']))

    # Quick terminal output check of whether the accuracy of the model is acceptable
    val_loss, val_acc = model.evaluate(X_test, Y_test)
    print(f'loss: {val_loss}, acc: {val_acc}')

    # Let the model predict the test values then round the numbers.
    model.predict(X_test)
    np.round(model.predict(X_test), 0)

    # Measure by which the model's metrics below are evaluated
    Y_pred = np.round(model.predict(X_test), 0)

    # Evaluation metrics that are saved to a .csv file written through a command line argument.
    mean_absolute_error = metrics.mean_absolute_error(Y_test, Y_pred)
    mean_squared_error = metrics.mean_squared_error(Y_test, Y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    r2_score = metrics.r2_score(Y_test, Y_pred)
    explained_variance_score = metrics.explained_variance_score(Y_test, Y_pred)
    d2_absolute_error_score = metrics.d2_absolute_error_score(Y_test, Y_pred)

    file_name = 'metrics/regression_neuralNetwork.csv'  # name of the file that is being written to
    cmd_header = f'echo echo "mean_absolut_error,mean_squared_error,root_mean_squared_error,r2_score,explained_variance_score,d2_absolute_error_score" >> {file_name}'
    cmd_data = f'echo "{mean_absolute_error},{mean_squared_error},{root_mean_squared_error},{r2_score},{explained_variance_score},{d2_absolute_error_score}" >> {file_name}'
    os.system(cmd_header)
    os.system(cmd_data)


neural_network(-10_000_000, 10_000_000, 1_000_000)



