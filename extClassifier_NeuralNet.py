from sklearn.metrics import classification_report  # for classification report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # tested various scaler
from sklearn.model_selection import train_test_split  # for splitting the data into train and test data
from tensorflow.keras import Sequential  # the model used
from tensorflow.keras.layers import Dense  # the main layers of the neural net
from tensorflow.keras.callbacks import EarlyStopping  # to stop the training of the model if over-fitting
import DataSet_Cleaner as dsc  # calling the right dataset to be used in this module
import tensorflow as tf  # importing the flattening layer
import numpy as np  # for rounding functions
import pandas as pd  # to convert classification report into a .csv file


def neural_network_extended_classification():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This is the neural network module used for the extended classification.

    """
    df = dsc.ml_algo_selection('extended_classifier')  # Calling the dataset for the extended classification.
    fund_flow_cats = df.filter(regex='fund_flow')  # filtering all the fund flow categories
    fund_flow_cats = pd.get_dummies(fund_flow_cats, columns=['fund_flow'])  # the fund flow categories to be predicted
    df = df.drop(['fund_flow'], axis=1)  # dropping predicted variable from the predictors

    X = df  # predicting variables
    y = fund_flow_cats  # predicted variable

    col_len = len(list(X.columns[:]))  # the number of predicting variables

    # Splitting the test and the training dataset in order to eliminate bias.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = MinMaxScaler()  # calling the scaling class from sklearn
    X_train = scaler.fit_transform(X_train)  # fitting the training data to the scaler
    X_test = scaler.transform(X_test)  # transforming the testing data to have the same shape as the training data

    epochs = 1500  # The epochs can be set arbitrarily high as the early stopping mechanism deciding on when to stop.

    # Instantiating the model and creating the nodes within the model
    model = Sequential()
    model.add(tf.keras.layers.Flatten())  # flattening the input gives slightly better results.
    # Input Layer
    model.add(Dense(col_len, input_shape=(X.shape[1],), kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(col_len, activation='relu'))  # hidden layer
    model.add(Dense(col_len, activation='relu'))  # hidden layer
    model.add(Dense(round(col_len/2, 0), activation='relu'))  # hidden layer
    model.add(Dense(20, activation='sigmoid'))  # output layer with sigmoid activation function acc. to best practice
    # Compiling model with the goal of accuracy
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Instantiating the early stopping class of sklearn
    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       patience=4,
                       restore_best_weights=True)

    # Callbacks allow for the early stopping of a model if over-fitting is detected.
    model.fit(x=X_train, y=Y_train, callbacks=[es], batch_size=10,
              epochs=epochs, validation_data=(X_test, Y_test), verbose=1)

    val_loss, val_acc = model.evaluate(X_test, Y_test)
    print(f'loss: {val_loss}, acc: {val_acc}')

    model.predict(X_test)  # predicting the future values

    Y_pred = np.round(model.predict(X_test), 0)  # rounding the predictions

    # Creating the classification report and converting it into a dataframe object to be exported as .csv file
    report = classification_report(Y_test, Y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('ext_report_nn.csv')


neural_network_extended_classification()




