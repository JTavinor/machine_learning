import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reads a csv and returns a pandas data frame
dataset = pd.read_csv('data/Data.csv')
# Matrix of features/independent variables, these are the data that we analyse to try and predict an output
# iloc selects certain data (rows, columns, cells) from a pandas data frame
# The values property turns the data frame back into a numpy array
X = dataset.iloc[:, :-1].values 

# Dependant variable vector, this is the data that we will be trying to predict
y = dataset.iloc[:, -1].values
# print(X, y)

# For missing data we can either remove the row, if the rows with missing data is a small percentage of the data set
# Or we can replace the missing data with the average of the column
# An Imputer is a tool used to handle missing data in a dataset, it can replace the missing data in different ways (i.e. mean, mode, median of the values in the column)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# print(X[:, 1:])

# fit fits the imputer on the data, this computes the necessary statistics for each column
# transform applies the imputation to the data (Can use fit_transform to combine these operations)
X[:, 1:] = imputer.fit(X[:, 1:]).transform(X[:, 1:])

# print(X)

# Use One-hot encoding for categorical (non-numerical) information - if we covert to numbers, the MLM may infer an order
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=1)
# print(X_train, X_test, y_train, y_test)

# Feature scaling - prevents the dataset being dominated by certain data
# Can do Standardisation or Normalisation
# Normalisation - best used when the data follows a normal distribution

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[: ,3:])
X_test[:, 3:] = sc.transform(X_test[: ,3:])

print(X_train)
print(X_test)