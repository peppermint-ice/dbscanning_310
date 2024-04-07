import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from config import paths
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

multivars = {
    'Parameter_name': [],
    'Parameter_value': [],
    'Correlating_parameter': [],
    'Model': [],
    'R2_score': [],
}


# Assuming df is your dataframe with 12 columns and 'target_column' is the column you want to predict
# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target_column', axis=1), df['target_column'], test_size=0.2, random_state=42)

best_score = float('inf')
best_features = None

# Iterate through all possible combinations of features
for r in range(1, len(df.columns)):
    for combination in itertools.combinations(df.columns.drop('target_column'), r):
        # Fit a model using the selected features
        model = LinearRegression()
        model.fit(X_train[list(combination)], y_train)
        # Evaluate the model on the testing set
        y_pred = model.predict(X_test[list(combination)])
        score = mean_squared_error(y_test, y_pred)
        # Update the best score and features if the current combination performs better
        if score < best_score:
            best_score = score
            best_features = combination

print("Best combination of features:", best_features)
print("Best MSE score:", best_score)