import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from load_sets import load_train_test_sets
from load_sets import load_train_test_sets_elaborate

from config import paths




if __name__ == '__main__':
    # Get the file path from the command-line argument
    file_path = sys.argv[1] ## /reconstruction_by_parameters

    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["ml_results"]

    # Read the CSV file specified by the command-line argument
    df = pd.read_csv(file_path)

    print(df['parameter_type'].unique())

    # First run of train-test split to set the desired columns
    X_train, X_test, y_train, y_test = load_train_test_sets(df, by_year=True)

    keys = [
        'Parameter_name',
        'Parameter_value',
        'Regression_model',
        'Correlating_parameter',
        'RMSE_score_calibration',
        'RMSE_score_validation',
        'R2_score_calibration',
        'R2_score_validation',
        'Successful_reconstructions_test',
        'Successful_reconstructions_train']
    current_results = dict.fromkeys(keys)
    results_svm = pd.DataFrame()
    try:
        print('starting grid search')
        # Define the grid of hyperparameters
        param_grid = {
            'C': [0.1, 1, 10],  # Penalty parameter C of the error term
            'epsilon': [0.01, 0.1, 0.3],  # Epsilon in the epsilon-SVR model
            'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
            'degree': [1, 2, 3, 4, 5],  # Degree of the polynomial kernel function
            'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        }

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters found by grid search
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Refactor model training to use the best hyperparameters
        model = SVR(**best_params)
        model.fit(X_train, y_train)

        pred_cal = model.predict(X_train)
        pred_val = model.predict(X_test)
        mse_cal = mean_squared_error(y_train, pred_cal)
        mse_val = mean_squared_error(y_test, pred_val)
        r2_cal = r2_score(y_train, pred_cal)
        r2_val = r2_score(y_test, pred_val)

        # print(parameter_type, ': ', parameter_value)
        # print('Correlating parameter: ', column)
        # print('r2 cal: ', r2_cal)
        # print('r2 val: ', r2_val)
        # print('')

        parameter_value = df['parameter_value'].unique()[0]
        parameter_type = df['parameter_type'].unique()[0]

        current_results['Parameter_value'] = parameter_value
        current_results['Parameter_name'] = parameter_type
        current_results['Regression_model'] = 'SVM'
        current_results['RMSE_score_calibration'] = mse_cal
        current_results['RMSE_score_validation'] = mse_val
        current_results['R2_score_calibration'] = r2_cal
        current_results['R2_score_validation'] = r2_val
        current_results['Successful_reconstructions_test'] = len(X_test)
        current_results['Successful_reconstructions_train'] = len(X_train)
        results_svm = pd.concat([results_svm, pd.DataFrame([current_results])], ignore_index=True)
        print(results_svm.shape)
        output_file = str(parameter_value) + parameter_type + '_results_svm.csv'
        output_file_path = os.path.join(csv_folder_path, output_file)
        results_svm.to_csv(output_file_path, index=False)
    except ValueError:
        print('A small dataset. Cannot calculate')
