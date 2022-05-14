import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def ml_algo_selection(ml_type):
    if ml_type == 'regression':
        data = pd.read_csv('data/Morningstar_data_version_5.0_lagged.csv')
        data.drop(list(data.filter(regex='Unnamed')), axis=1, inplace=True)
        data.drop(['Management Company', 'Name', 'Inception \nDate'], axis=1, inplace=True)

        return data

    elif ml_type == 'classifier':
        data = pd.read_csv('data/Morningstar_data_version_5.0_lagged.csv')
        data.drop(list(data.filter(regex='Unnamed')), axis=1, inplace=True)

        def ff_positive(x):
            if x >= 0.0:
                return 1
            elif x < 0.0:
                return 0

        data['fund_flow'] = data['fund_flow'].apply(ff_positive)
        data.drop(['Management Company', 'Name', 'Inception \nDate'], axis=1, inplace=True)

        return data


def svm_classification():
    df = ml_algo_selection('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = svm.SVC(kernel='linear', verbose=1)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    classi = classification_report(Y_test, Y_pred)

    print(classi)


# svm_classification()


def k_nearest_neightbour():
    df = ml_algo_selection('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, algorithm='brute', metric='cosine')
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    classi = classification_report(Y_test, Y_pred)

    print(classi)


# k_nearest_neightbour()


def random_forrest():
    df = ml_algo_selection('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(verbose=1, n_jobs=-1, n_estimators=100)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    classi = classification_report(Y_test, Y_pred)

    print(classi)


random_forrest()


def random_forrest2():
    df = ml_algo_selection('classifier')

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

    grid_search = GridSearchCV(estimator=rf,
                               param_grid=params,
                               cv=4, verbose=1, scoring="accuracy")

    grid_search.fit(X_train, Y_train)

    print(grid_search.best_params_)
    grid_predictions = grid_search.predict(X_test)

    print(classification_report(Y_test, grid_predictions))

    best_score = grid_search.best_score_
    print(best_score)
    rf_best = grid_search.best_estimator_
    importance = rf_best.feature_importances_

    imp_df = pd.DataFrame({
        "Varname": X_train.columns,
        "Imp": importance
    })
    print(imp_df.sort_values(by="Imp", ascending=False))


def linear_regression():
    df = ml_algo_selection('classifier')

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
