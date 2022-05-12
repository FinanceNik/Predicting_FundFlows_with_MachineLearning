import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt


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


def linear_regression():
    df = regression_type_data('classifier')

    predictor = 'fund_flow'
    drops = [predictor]

    X = df.drop(drops, axis=1).values
    y = df[predictor].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    print(f' R^2: {r2_score(Y_test, Y_pred)}')
