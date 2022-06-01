import pandas as pd  # used for creating a dataframe object to store a classification report
from sklearn.model_selection import train_test_split  # splitting the data into training and test sets
from sklearn.preprocessing import MinMaxScaler  # scaling the data according to best practice
from sklearn.ensemble import RandomForestClassifier  # the model used in this module
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # evaluation measures
from sklearn.model_selection import GridSearchCV  # used for hyperparameter tuning
import Statistics  # calling visualization functions created in this module
import DataSet_Cleaner as dsc  # retrieving the correct dataset for this model's aim


def random_forrest_binary_classification():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This module creates the random forest model for binary classification. Here, it is aimed at predicting whether a
    fund flow in the future is positive or negative.

    """
    df = dsc.ml_algo_selection('classifier')  # retrieving the dataset for the binary classification

    predictor = 'fund_flow'  # predicting fund flows
    drops = [predictor]

    # Mapping the corresponding features to predictors and predicted.
    X = df.drop(drops, axis=1).values  # All the predictors
    y = df[predictor].values  # The predicted

    # Splitting the test and the training dataset in order to eliminate bias.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data to make it easier for the ML model to work with the data.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Calling the actual model
    model = RandomForestClassifier(verbose=1, n_jobs=-1, n_estimators=100)
    # Fitting the data to the model
    model.fit(X_train, Y_train)

    # Predicting fund flows
    Y_pred = model.predict(X_test)
    # Testing the accuracy
    accu = accuracy_score(Y_test, Y_pred)
    # Creating the confusion matrix
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    # Creating the classification report
    report = classification_report(Y_test, Y_pred, output_dict=True)
    # Retrieving the feature importance for the model
    feature_imp = model.feature_importances_
    # Mapping the importance of the features to their corresponding names
    feature_names = list(df.drop(drops, axis=1).columns[:])

    # Retrieving the visualization functions that have been coded in the Statistics module
    # Feature importance visualization
    Statistics.feature_importance(feature_names, feature_imp, 'Random Forest')
    # Confusion matrix visualization
    Statistics.confusion_matrix(conf_matrix, 'Random Forest')

    # Convert the classification report to a dataframe and then exporting that dataframe to a .csv
    df = pd.DataFrame(report).transpose()
    df.to_csv('rf_report.csv')

    # Printing the accuracy for quick access
    print(accu)


random_forrest_binary_classification()


def random_forrest_hyperparameter_tuning():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Here, the hyperparameters for the random forest model are tuned and chosen.

    """
    df = dsc.ml_algo_selection('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(verbose=1, random_state=42, n_jobs=-1)
    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }

    # Calling the GridSearchCV function and fitting the model.
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=params,
                               cv=4, verbose=1, scoring="accuracy")

    # Fitting the data to the model.
    grid_search.fit(X_train, Y_train)

    # Printing the best parameters.
    print(grid_search.best_params_)
    grid_predictions = grid_search.predict(X_test)
    # Printing the best score of the model.
    best_score = grid_search.best_score_
    print(best_score)
