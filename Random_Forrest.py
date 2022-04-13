from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import Data_Handler as dh

data = dh.df

# target
y = data['Darlehen - Vertragsstatus']


# load X variables into a pd df with columns
X = data.drop(['Darlehen - Vertragsstatus'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

# initiate model
rfc = RandomForestClassifier(n_estimators=40, verbose=1, n_jobs=10)

# give the model data
rfc.fit(X_train, y_train)

# predict
predictions = rfc.predict(X_test)
prediction_rate = confusion_matrix(y_test, predictions)[1][1] / \
                  (confusion_matrix(y_test, predictions)[1][1] + confusion_matrix(y_test, predictions)[1][0])

print(confusion_matrix(y_test, predictions), f'prediction accuracy rate: {round(prediction_rate*100, 1)}%')
print('\n')
print(classification_report(y_test, predictions))




