import pandas as pd  # used for the handling of data files like .csv
from sklearn.model_selection import train_test_split  # for splitting the data into training and test sets
from sklearn.preprocessing import MinMaxScaler  # for scaling the data and normalizing it
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # for evaluating the model
from sklearn.ensemble import GradientBoostingClassifier  # the Gradient Boosting Classifier model used
import Statistics  # for the statistics and visualizations of the model
import DataSet_Cleaner as dsc  # for the data cleaning of the data and retrieving the appropriate dataset


def gradient_boosting():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This is the binary classification function for the gradient boosting machine learning model.

    """
    df = dsc.ml_algo_selection('classifier')
    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    # Splitting the test and the training dataset in order to eliminate bias.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the values because it makes it easier for ML models to work with them.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Instantiating the Gradient Boosting Classifier model.
    model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.05, max_depth=3,
                                       subsample=0.5, validation_fraction=0.1, n_iter_no_change=20,
                                       max_features='log2', verbose=1)
    # Fit the model to the training data.
    model.fit(X_train, Y_train)

    # Predicting the test set results.
    Y_pred = model.predict(X_test)
    accu = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    feature_imp = model.feature_importances_

    # Mapping the feature names to the feature importance values.
    feature_names = list(df.drop(drops, axis=1).columns[:])

    # Plotting the feature importance and the confusion matrix.
    Statistics.feature_importance(feature_names, feature_imp, 'Gradient Boosting')
    Statistics.confusion_matrix(conf_matrix, 'Gradient Boosting')

    # Creating and saving the classification report to a .csv file.
    report = classification_report(Y_test, Y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('gb_report.csv')

    # Plotting the accuracy of the model.
    print(accu)


gradient_boosting()