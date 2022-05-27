import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import Statistics
import DataSet_Cleaner as dsc


def random_forrest():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------


    """
    df = dsc.ml_algo_selection('extended_classifier')
    df = df.sample(500_000)

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(verbose=1, n_jobs=1, n_estimators=100)
    model.fit(X_train, Y_train)

    # save the model to disk
    # filename = 'ext_rf_model.sav'
    # pickle.dump(model, open(filename, 'wb'))

    # # retrieve the model
    # model = pickle.load(open(filename, 'rb'))

    Y_pred = model.predict(X_test)
    accu = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True)  # has to be true for .csv conversion
    feature_imp = model.feature_importances_

    feature_names = list(df.drop(drops, axis=1).columns[:])

    Statistics.feature_importance(feature_names, feature_imp, 'Extended Random Forest Classifier')
    Statistics.confusion_matrix(conf_matrix, 'Extended Random Forest Classifier')

    df = pd.DataFrame(report).transpose()

    df.to_csv('ext_rf_report.csv')

    print(accu)
    print(report)


random_forrest()


def random_forrest2():
    df = dsc.ml_algo_selection('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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

