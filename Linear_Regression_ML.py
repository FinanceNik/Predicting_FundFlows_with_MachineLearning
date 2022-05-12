import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from matplotlib import pyplot as plt

# print(list(pd.read_csv('data/Morningstar_data_version_4.0.csv').columns[:]), '\n')


def regression_type_data(ml_type):
    if ml_type == 'regression':
        data = pd.read_csv('data/Morningstar_data_version_4.0.csv')
        data.drop(list(data.filter(regex='Unnamed')), axis=1, inplace=True)
        data.drop(['Management Company', 'Name', 'Inception \nDate'], axis=1, inplace=True)

        return data

    elif ml_type == 'classifier':
        data = pd.read_csv('data/Morningstar_data_version_4.0.csv')
        data.drop(list(data.filter(regex='Unnamed')), axis=1, inplace=True)

        def ff_positive(x):
            if x >= 0.0:
                return 1
            elif x < 0.0:
                return 0

        data['fund_flow'] = data['fund_flow'].apply(ff_positive)
        data.drop(['Management Company', 'Name', 'Inception \nDate'], axis=1, inplace=True)

        return data


def random_forrest():
    df = regression_type_data('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(verbose=1, n_jobs=40, n_estimators=40)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    accu = accuracy_score(Y_test, Y_pred)
    classi = classification_report(Y_test, Y_pred)

    print(accu, '\n', classi)


# random_forrest()


def random_forrest2():
    df = regression_type_data('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }

    grid_search = GridSearchCV(estimator=rf,
                               param_grid=params,
                               cv=4,
                               n_jobs=-1, verbose=1, scoring="accuracy")

    grid_search.fit(X_train, Y_train)

    best_score = grid_search.best_score_
    rf_best = grid_search.best_estimator_

    print(best_score)


random_forrest2()


def linear_regression():
    df = regression_type_data('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    print(f' R^2: {r2_score(Y_test, Y_pred)}')


# linear_regression()
