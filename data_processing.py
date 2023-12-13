import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


dataset = pd.read_csv('data/Data.csv')
x = dataset.iloc[:, :-1].values # Matrix of features/independent variables
y = dataset.iloc[:, -1].values # Dependant variable vector
print(x, y)

# For missing data we can either remove the row, if the rows with missing data is a small percentage of the data set
# Or we can replace the missing data with the average of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
print(x[:, 1:])

x[:, 1:] = imputer.fit(x[:, 1:]).transform(x[:, 1:])

print(x)

# Use One-hot encoding for categorical (non-numerical) information - if we covert to numbers, the MLM may infer an order
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)
