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
import Statistics
import pickle
import DataSet_Cleaner as dsc


def svm_classification():
    df = dsc.ml_algo_selection('classifier')

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
    df = dsc.ml_algo_selection('classifier')

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
    df = dsc.ml_algo_selection('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(verbose=1, n_jobs=-1, n_estimators=100)
    model.fit(X_train, Y_train)

    # save the model to disk
    filename = 'rf_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # # retrieve the model
    # model = pickle.load(open(filename, 'rb'))

    Y_pred = model.predict(X_test)
    accu = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True)
    feature_imp = model.feature_importances_

    feature_names = list(df.drop(drops, axis=1).columns[:])

    Statistics.feature_importance(feature_names, feature_imp, 'Random Forest')
    Statistics.confusion_matrix(conf_matrix, 'Random Forest')

    df = pd.DataFrame(report).transpose()

    df.to_csv('report.csv')

    print(accu)


# random_forrest()


def random_forrest2():
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
    df = dsc.ml_algo_selection('classifier')

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
