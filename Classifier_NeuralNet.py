from sklearn.model_selection import train_test_split  # for splitting data into train and test sets
from sklearn.metrics import classification_report, confusion_matrix  # for evaluating the conf. matrix and class. report
from sklearn.preprocessing import MinMaxScaler  # scaling the data
from tensorflow.keras import Sequential  # the model used in the Neural Network
from tensorflow.keras.layers import Dense, Dropout  # The layers of the model used
import tensorflow as tf  # for the GPU and calling the flattening function
import pandas as pd  # for the dataframe object that is saving the classification report
from tensorflow.keras.callbacks import EarlyStopping  # for early stopping the model if overtraining is detected
import DataSet_Cleaner as dsc  # retrieving the correct dataset
import Statistics  # Visualization functions
import numpy as np  # for the rounding values of the confusion matrix and the classification report


def neural_network_classification():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This is the neural network module used for the binary classification.

    """
    df = dsc.ml_algo_selection('classifier')  # Calling binary classification variant of the dataset.
    col_len = len(df.drop(['fund_flow'], axis=1).columns[:])

    predictor = 'fund_flow'  # of course, the goal is to predict fund flows.
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    # Splitting the test and the training dataset in order to eliminate bias.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Scaling the data.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Defining the number of epochs. Will be stopped by early stopping, hence number can be very high.
    epochs = 800

    # Creating the model and adding the layers.
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

    # Early stopping for overfitting.
    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       patience=8,
                       restore_best_weights=True)
    # Fitting the modela and determine the parameters of the model.
    hist = model.fit(x=X_train, y=Y_train, callbacks=[es], batch_size=10,
                     epochs=epochs, validation_data=(X_test, Y_test), verbose=1)  # include batch size

    # Evaluating the model visually.
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # Visualizing the training and validation loss.
    Statistics.loss_visualizer(train_loss, val_loss, len(hist.history['loss']))
    Statistics.accuracy_visualizer(train_acc, val_acc, len(hist.history['loss']))

    # Show basic metrics quickly in terminal.
    val_loss, val_acc = model.evaluate(X_test, Y_test)
    print(f'loss: {val_loss}, acc: {val_acc}')

    # Predicting the test set.
    model.predict(X_test)
    np.round(model.predict(X_test), 0)

    preds = np.round(model.predict(X_test), 0)  # rounding the values for better visualization

    # Calling the confusion matrix function.
    Statistics.confusion_matrix(confusion_matrix(Y_test, preds), 'Neural Network Classification')

    # Creating the classification report.
    report = classification_report(Y_test, preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('nn_report.csv')

    # The feature importance cannot be visualized for neural networks, unfortunately. It was tried anyway and put inside
    # a try-and-except statement so the module does not throw an error and stop.
    try:
        feature_imp = model.feature_importances_
        feature_names = list(df.drop(drops, axis=1).columns[:])
        Statistics.feature_importance(feature_names, feature_imp, 'Neural Net for Classification')
    except:
        pass