import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target, feature):
    X = df[[feature]]
    y = df[target]
    return train_test_split(X, y, test_size=0.2)

def impute_data(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed, imputer

def train_model(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(regressor, X_test, y_test):
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def predict_new_data(regressor, imputer, new_data):
    new_data_imputed = imputer.transform(new_data)
    return regressor.predict(new_data_imputed)

def plot_data(df, X_train, y_train, X_test, y_test, regressor, imputer):
    feature = df.columns[0]  # Assuming the first column is the feature
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='red', label='Test data')

    x_line = np.linspace(df[feature].min(), df[feature].max(), 100).reshape(-1, 1)
    y_line = regressor.predict(imputer.transform(x_line))
    plt.plot(x_line, y_line, color='green', label='Regression Line')

    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title('Regression Analysis')
    plt.legend()
    plt.show()


def print_regression_equation(regressor):
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    print("Linear Regression Equation: y = {:.2f} * x + {:.2f}".format(slope, intercept))

# Main Execution
df = load_data('linear_regression_practice/data.csv')

X_train, X_test, y_train, y_test = preprocess_data(df, 'Price', 'Size')
X_train_imputed, X_test_imputed, imputer = impute_data(X_train, X_test)
regressor = train_model(X_train_imputed, y_train)

mse, r2 = evaluate_model(regressor, X_test_imputed, y_test)
print("Mean Squared Error: ", mse)
print("R-squared: ", r2)

new_data = pd.DataFrame({'Size': [60, 70, 80]})
new_predictions = predict_new_data(regressor, imputer, new_data)

for size, price in zip(new_data['Size'], new_predictions):
    print(f"Predicted Price for Size {size}: {price}")

plot_data(df, X_train, y_train, X_test, y_test, regressor, imputer)

print_regression_equation(regressor)
