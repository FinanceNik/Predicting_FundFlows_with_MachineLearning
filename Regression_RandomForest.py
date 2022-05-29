from sklearn.model_selection import train_test_split  # for splitting the dataset into test and train
from sklearn.preprocessing import MinMaxScaler  # scaling the data
from sklearn.ensemble import RandomForestRegressor  # model used
from sklearn.model_selection import GridSearchCV  # for hyperparameter tuning
import DataSet_Cleaner as dsc  # including visualization functions that are coded in this module
import numpy as np  # for higher math functions
from sklearn import metrics  # evaluating the model
import Statistics
import os  # writing data to files through command line arguments


def random_forrest_regressor(min, max, n):
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This is the machine learning random forest model used for the prediction of the actual values of the future fund
    flows. The [min, max, n] arguments stand for:
                                                  - minimum fund flow to be included
                                                  - maximum fund flow to be included
                                                  - number of the sample size
    These limitations are necessary because the fund flow variance is too high and the models give horrible
    predictions. Hence, in order to homogenize the data a bit, the outliers are excluded.

    """
    df = dsc.ml_algo_selection('regression')
    df = df[(df["fund_flow"] < max)]  # maximum fund flow value to be included
    df = df[(df["fund_flow"] > min)]  # minimum fund flow value to be included
    df = df.sample(n)  # sample size to be used

    # Scaling the fund flows from -1 to 1 according to the practice described in the paper.
    def scaling(x):
        x = (2 * (x - min) / (max - min)) - 1  # mathematical scaling function
        return x

    df['fund_flow'] = df['fund_flow'].apply(scaling)  # applying the scaling

    predictor = 'fund_flow'  # predicting fund flows of course
    drops = [predictor]  # dropping fund flow from the predictors

    X = df.drop(drops, axis=1).values  # predicting variables
    y = df[predictor].values  # predictor variable

    # decreasing bias through splitting of test and training data.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Scaling the data as ML models can work better with scaled data.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Calling the model and fitting the training data to it
    model = RandomForestRegressor(verbose=1, n_jobs=10, n_estimators=1000)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    mean_absolute_error = metrics.mean_absolute_error(Y_test, Y_pred)
    mean_squared_error = metrics.mean_squared_error(Y_test, Y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    r2_score = metrics.r2_score(Y_test, Y_pred)
    explained_variance_score = metrics.explained_variance_score(Y_test, Y_pred)
    d2_absolute_error_score = metrics.d2_absolute_error_score(Y_test, Y_pred)

    # Writing the regression metrics to a .csv file through command line executions.
    file_name = 'metrics/regression_randomForest.csv'
    cmd_header = f'echo "mean_absolut_error,mean_squared_error,root_mean_squared_error,r2_score,explained_variance_score,d2_absolute_error_score" >> {file_name}'
    cmd_data = f'echo "{mean_absolute_error},{mean_squared_error},{root_mean_squared_error},{r2_score},{explained_variance_score},{d2_absolute_error_score}" >> {file_name}'
    os.system(cmd_header)
    os.system(cmd_data)

    try:
        feature_imp = model.feature_importances_
        feature_names = list(df.drop(drops, axis=1).columns[:])
        Statistics.feature_importance(feature_names, feature_imp, 'Random Forest Regressor')
    except:
        pass


random_forrest_regressor(-10_000_000, 10_000_000, 1_000_000)


def random_forrest_regressor_hyperparameter_tuning(min, max, n):
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This module is used for the tuning of the hyperparameters of the random forest algorithm for regression purposes.

    """
    df = dsc.ml_algo_selection('regression')  # retrieving the correct dataset for regression
    df = df[(df["fund_flow"] < max)]  # maximum fund flow value to be included
    df = df[(df["fund_flow"] > min)]  # minimum fund flow value to be included
    df = df.sample(n)  # sample size to be used

    # Scaling the fund flows from -1 to 1 according to the practice described in the paper.
    def scaling(x):
        x = (2 * (x - min) / (max - min)) - 1
        return x

    df['fund_flow'] = df['fund_flow'].apply(scaling)  # retrieving the extended classification dataset.

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(verbose=1, n_jobs=10, n_estimators=1000)
    model.fit(X_train, Y_train)

    # The parameters that are to be tested
    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }

    # calling the grid search cross validation function
    grid_search = GridSearchCV(estimator=model,
                               param_grid=params,
                               cv=4, verbose=1, scoring="accuracy")
    # Fitting the training data to the grid search cv algorithm
    grid_search.fit(X_train, Y_train)

    # Show the best parameters
    print(grid_search.best_params_)
    grid_predictions = grid_search.predict(X_test)
    print(grid_predictions)
    best_score = grid_search.best_score_
    print(best_score)
