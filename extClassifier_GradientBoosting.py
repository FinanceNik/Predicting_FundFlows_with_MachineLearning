import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # data is split into train/test
from sklearn.preprocessing import MinMaxScaler  # scaling the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # evaluation of the model
from sklearn.ensemble import GradientBoostingClassifier  # the model used in this module
import Statistics  # for visualizations
import DataSet_Cleaner as dsc  # Retrieving the correct dataset


def gradient_boosting_extended_classification():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This module inherits the gradient boosting algorithm for the extended classification.

    """
    df = dsc.ml_algo_selection('extended_classifier')  # Calling the extended classification dataset
    predictor = 'fund_flow'  # Wanting to predict fund flows
    drops = [predictor]  # Dropping the fund flows from the predicting variables

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    # Splitting the test and the training dataset in order to eliminate bias.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data as it makes it easier for ML models to work with them.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Instantiating the gradient boosting classifier.
    model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.05, max_depth=3, subsample=0.5,
                                       validation_fraction=0.1, n_iter_no_change=20, max_features='log2',
                                       verbose=1)
    # Fitting the data to the model.
    model.fit(X_train, Y_train)

    # Predicting the out-of-sample data.
    Y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    # Modelling of the feature importance as well as the confusion matrix and visualizing them.
    feature_imp = model.feature_importances_
    feature_names = list(df.drop(drops, axis=1).columns[:])
    Statistics.feature_importance(feature_names, feature_imp, 'Gradient Boosting Model for Extended Classification')
    Statistics.confusion_matrix(conf_matrix, 'Gradient Boosting Model for Extended Classification')

    # Classification report.
    report = classification_report(Y_test, Y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('ext_gb_report.csv')

    # Printing the accuracy score.
    accu = accuracy_score(Y_test, Y_pred)
    print(accu)


gradient_boosting_extended_classification()