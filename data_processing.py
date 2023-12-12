import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/Data.csv')
x = dataset.iloc[:, :-1].values # Matrix of features/independent variables
y = dataset.iloc[:, -1].values # Dependant variable vector
print(x)
print(y)