import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import Statistics
import pickle
import DataSet_Cleaner as dsc
import numpy as np
from sklearn import metrics
import os


def gradient_boosting_for_regression(min, max, n):
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This is the machine learning gradient boosting model used for the prediction of the actual values of the future fund
    flows. The [min, max, n] arguments stand for:
                                                  - minimum fund flow to be included
                                                  - maximum fund flow to be included
                                                  - number of the sample size
    These limitations are necessary because the fund flow variance is too high and the models give horrible
    predictions. Hence, in order to homogenize the data a bit, the outliers are excluded. The number of samples had to
    be given for testing purposes, as running the algo for testing on the whole dataset is time costly.

    """
    df = dsc.ml_algo_selection('regression')  # Calling the regression type of the dataset
    df = df[(df["fund_flow"] < max)]  # applying the maximum filter
    df = df[(df["fund_flow"] > min)]  # applying the minimum filter
    df = df.sample(n)  # taking a number of samples according to the n defined

    def scaling(x):
        x = (2 * (x - min) / (max - min)) - 1
        return x

    df['fund_flow'] = df['fund_flow'].apply(scaling)  # applying the -1 to 1 scaling function on the fund flows

    predictor = 'fund_flow'  # predicting fund flows
    drops = [predictor]  # dropping fund flows

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05, max_depth=3, subsample=0.5,
                                       validation_fraction=0.1, n_iter_no_change=20, max_features='log2', verbose=1)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    mean_absolute_error = metrics.mean_absolute_error(Y_test, Y_pred)
    mean_squared_error = metrics.mean_squared_error(Y_test, Y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    r2_score = metrics.r2_score(Y_test, Y_pred)
    explained_variance_score = metrics.explained_variance_score(Y_test, Y_pred)
    d2_absolute_error_score = metrics.d2_absolute_error_score(Y_test, Y_pred)

    file_name = 'metrics/regression_gradientBoosting.csv'
    cmd_header = f'echo echo "mean_absolut_error,mean_squared_error,root_mean_squared_error,r2_score,explained_variance_score,d2_absolute_error_score" >> {file_name}'
    cmd_data = f'echo "{mean_absolute_error},{mean_squared_error},{root_mean_squared_error},{r2_score},{explained_variance_score},{d2_absolute_error_score}" >> {file_name}'
    os.system(cmd_header)
    os.system(cmd_data)

    try:
        feature_imp = model.feature_importances_
        feature_names = list(df.drop(drops, axis=1).columns[:])
        Statistics.feature_importance(feature_names, feature_imp, 'Gradient Boosting Regressor')
    except:
        pass


