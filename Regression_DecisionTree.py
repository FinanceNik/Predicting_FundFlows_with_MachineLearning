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
import os


def random_forrest():
    df = dsc.ml_algo_selection('regression')
    # df['fund_flow'] = np.log(df['fund_flow'])  # If nothing works, try logging values.
    df = df.sample(1_000)
    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(verbose=1, n_jobs=10, n_estimators=1000)
    model.fit(X_train, Y_train)

    filename = 'rf_reg_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # # retrieve the model
    # model = pickle.load(open(filename, 'rb'))

    Y_pred = model.predict(X_test)

    mean_absolute_error = metrics.mean_absolute_error(Y_test, Y_pred)
    mean_squared_error = metrics.mean_squared_error(Y_test, Y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    r2_score = metrics.r2_score(Y_test, Y_pred)

    try:
        explained_variance_score = metrics.explained_variance_score(Y_test, Y_pred)
        print('Explained Variance:', explained_variance_score)
    except:
        explained_variance_score = 'none'

    try:
        d2_absolute_error_score = metrics.d2_absolute_error_score(Y_test, Y_pred)
        print('D2 Absolut Error:', d2_absolute_error_score)
    except:
        d2_absolute_error_score = 'none'

    file_name = 'regression_metrics/regression_randomForest.csv'
    cmd_header = f'echo echo "mean_absolut_error,mean_squared_error,root_mean_squared_error,r2_score,explained_variance_score,d2_absolute_error_score" >> {file_name}'
    cmd_data = f'echo "{mean_absolute_error},{mean_squared_error},{root_mean_squared_error},{r2_score},{explained_variance_score},{d2_absolute_error_score}" >> {file_name}'
    os.system(cmd_header)
    os.system(cmd_data)

    try:
        feature_imp = model.feature_importances_
        feature_names = list(df.drop(drops, axis=1).columns[:])
        Statistics.feature_importance(feature_names, feature_imp, 'Random Forest Regressor')
    except:
        pass

    # Re-run this dog shit, r-squared == -0.0453


