from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import Data_Handler as dh

df = dh.data_cleaning()
X = df.drop(['Darlehen - Vertragsstatus'], axis=1).values
y = df['Darlehen - Vertragsstatus'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(26, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, epochs=10, validation_data=(X_test, y_test))

predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()


