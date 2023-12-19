import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target, features):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2)

def impute_data(X_train, X_test):
    mean_imputer = SimpleImputer(strategy='mean')
    mode_imputer = SimpleImputer(strategy='most_frequent')

    preprocessor = ColumnTransformer(
        transformers=[
            ('mean_imputer', mean_imputer, ['Size', 'Age']),
            ('mode_imputer', mode_imputer, ['Bedrooms'])
        ])

    X_train_imputed = preprocessor.fit_transform(X_train)
    X_test_imputed = preprocessor.transform(X_test)

    return X_train_imputed, X_test_imputed, preprocessor


def train_model(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


df = load_data('multiple_linear_regression/multiple_linear_regression_dataset.csv')

X_train, X_test, y_train, y_test = preprocess_data(df, 'Price', ['Size', 'Bedrooms', 'Age'])
X_train_imputed, X_test_imputed, imputer = impute_data(X_train, X_test)
regressor = train_model(X_train_imputed, y_train)

mse, r2 = evaluate_model(regressor, X_test_imputed, y_test)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)