import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('data/Data.csv')
X = dataset.iloc[:, :-1].values # Matrix of features/independent variables
y = dataset.iloc[:, -1].values # Dependant variable vector
print(X, y)

# For missing data we can either remove the row, if the rows with missing data is a small percentage of the data set
# Or we can replace the missing data with the average of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
print(X[:, 1:])

X[:, 1:] = imputer.fit(X[:, 1:]).transform(X[:, 1:])

print(X)

# Use One-hot encoding for categorical (non-numerical) information - if we covert to numbers, the MLM may infer an order
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

le = LabelEncoder()
y = le.fit_transform(y)

print(y)


