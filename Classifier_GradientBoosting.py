import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import Statistics
import pickle
import DataSet_Cleaner as dsc


def gradient_boosting():
    df = dsc.ml_algo_selection('classifier')
    # df = df.sample(10_000)
    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.05, max_depth=3, subsample=0.5,
                                       validation_fraction=0.1, n_iter_no_change=20, max_features='log2', verbose=1)
    model.fit(X_train, Y_train)

    # save the model to disk
    filename = 'gb_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # # retrieve the model
    # model = pickle.load(open(filename, 'rb'))

    Y_pred = model.predict(X_test)
    accu = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True)
    feature_imp = model.feature_importances_

    feature_names = list(df.drop(drops, axis=1).columns[:])

    Statistics.feature_importance(feature_names, feature_imp, 'Gradient Boosting')
    Statistics.confusion_matrix(conf_matrix, 'Gradient Boosting')

    df = pd.DataFrame(report).transpose()

    df.to_csv('gb_report.csv')

    print(accu)


gradient_boosting()