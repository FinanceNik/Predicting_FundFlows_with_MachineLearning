import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.preprocessing import MinMaxScaler  # scaling the data
from sklearn.ensemble import RandomForestClassifier  # the model used
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # evaluating the model
from sklearn.model_selection import GridSearchCV  # grid search cross validation for hyperparameter tuning
import Statistics  # for visualizations
import DataSet_Cleaner as dsc  # retrieving the correct dataset


def random_forrest_extended_classification():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This module inherits the gradient boosting algorithm for the extended classification.

    """
    df = dsc.ml_algo_selection('extended_classifier')  # Retrieving the dataset for the extended classification

    predictor = 'fund_flow'  # Again, predicting fund flows
    drops = [predictor]  # Dropping fund flow variable from the predictors set

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    # Splitting the training and the testing data to eliminate bias.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Scaling the data according to best practice.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(verbose=1, n_jobs=1, n_estimators=100)  # Instantiating the model
    model.fit(X_train, Y_train)  # fitting the training data to the model

    Y_pred = model.predict(X_test)  # predicting the test values
    accu = accuracy_score(Y_test, Y_pred)  # calculating the accuracy score
    conf_matrix = confusion_matrix(Y_test, Y_pred)  # calling the confusion matrix

    # Calculating the feature importance for the random forest and mapping the names of the variables respectively
    feature_imp = model.feature_importances_
    feature_names = list(df.drop(drops, axis=1).columns[:])
    # Calling the visualization functions in the Statistics module to create plots of the feature imp. & conf_matrix
    Statistics.feature_importance(feature_names, feature_imp, 'Extended Random Forest Classifier')
    Statistics.confusion_matrix(conf_matrix, 'Extended Random Forest Classifier')

    # Calculating the classification report and converting it into a dataframe object to be saved as .csv file
    report = classification_report(Y_test, Y_pred, output_dict=True)  # output_dict has to be true for .csv conversion
    df = pd.DataFrame(report).transpose()  # fitting the transposed report to the dataframe object
    df.to_csv('ext_rf_report.csv')  # exporting the classification report as .csv

    print(accu)  # printing the accuracy for quick check


random_forrest_extended_classification()


def random_forrest_extended_classification_hyperparameter_tuning():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This module is used for the tuning of the hyperparameters of the random forest algorithm.

    """
    df = dsc.ml_algo_selection('extended_classifier')  # retrieving the extended classification dataset.

    predictor = 'fund_flow'  # predicting fund flow values
    drops = [predictor]  # dropping the fund flow variable from the predictors

    X = df.drop(drops, axis=1).values  # predicting variables
    y = df[predictor].values  # predictor variable

    # Splitting the dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scaling the data according to best practice
    scaler = MinMaxScaler()  # calling the scaling class
    X_train = scaler.fit_transform(X_train)  # fitting the training data
    X_test = scaler.transform(X_test)  # transforming the test data

    # Instantiating the Random Forest Classification Model
    model = RandomForestClassifier(verbose=1, random_state=42, n_jobs=-1)
    # Stating the parameters that are to be tested
    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }
    # Instantiating the grid search cross validation class and fitting the model as well as the parameters to it.
    grid_search = GridSearchCV(estimator=model,
                               param_grid=params,
                               cv=4, verbose=1, scoring="accuracy")  # scoring for accuracy as is most import. metric

    grid_search.fit(X_train, Y_train)  # fitting the training data

    print(grid_search.best_params_)
    grid_predictions = grid_search.predict(X_test)

    print(classification_report(Y_test, grid_predictions))

    best_score = grid_search.best_score_
    print(best_score)


