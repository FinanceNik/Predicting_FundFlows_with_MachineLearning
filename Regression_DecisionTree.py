import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import Statistics
import pickle
import DataSet_Cleaner as dsc
import numpy as np
from sklearn import metrics


def random_forrest():
    df = dsc.ml_algo_selection('regression')
    # df['fund_flow'] = np.log(df['fund_flow'])  # If nothing works, try logging values.
    df = df.sample(10_000)
    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(verbose=1, n_jobs=-1, n_estimators=1000)
    model.fit(X_train, Y_train)

    filename = 'rf_reg_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # # retrieve the model
    # model = pickle.load(open(filename, 'rb'))

    Y_pred = model.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    print('R-squared:', metrics.r2_score(Y_test, Y_pred))
    print('Explained Variance:', metrics.explained_variance_score(Y_test, Y_pred))
    print('D2 Absolut Error:', metrics.d2_absolute_error_score(Y_test, Y_pred))

    feature_imp = model.feature_importances_

    feature_names = list(df.drop(drops, axis=1).columns[:])

    Statistics.feature_importance(feature_names, feature_imp, 'Random Forest Regressor')

    # Re-run this dog shit, r-squared == -0.0453


random_forrest()