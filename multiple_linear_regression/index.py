import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ds = pd.read_csv('multiple_linear_regression/50_Startups.csv')

X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Probably need to drop one of the encoded categorical variables to avoid the dummy trap
# Building a model? I.e. Backward elimination

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
print("R^2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

