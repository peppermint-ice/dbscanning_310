import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from load_sets import load_train_test_sets

from config import paths


def preprocess_data(df, target_column='Measured_leaf_area', by_year=False):
    parameter_type = df['parameter_type'].unique()[0]
    parameter_value = df['parameter_value'].unique()[0]
    df.loc[:, 'Year'] = ('20' + df['File_name'].str[:6].str[-2:]).astype(int)
    df_train = df[df['Year'] == 2023]
    df_test = df[df['Year'] == 2024]
    if by_year:
        X_train = df_train.drop(columns=[target_column, 'File_name', 'parameter_type', 'parameter_value', 'Year'],
                                inplace=False, axis=1)
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column, 'File_name', 'parameter_type', 'parameter_value', 'Year'],
                              inplace=False, axis=1)
        y_test = df_test[target_column]
    else:
        df.drop(columns=['File_name', 'parameter_type', 'parameter_value', 'Year'],
                inplace=True, axis=1)
        # Define the number of quantiles
        num_quantiles = 10

        # Assuming df is your DataFrame
        # First, divide the DataFrame into quantiles based on the target column
        df['quantile'] = pd.qcut(df[target_column], q=num_quantiles, labels=False)

        # Initialize empty lists to store train and test sets
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []

        # Iterate over each quantile
        for quantile in range(num_quantiles):
            # Filter the DataFrame for the current quantile
            df_quantile = df[df['quantile'] == quantile]

            # Split the quantile into train and test sets
            X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(df_quantile.drop(target_column, axis=1),
                                                                        df_quantile[target_column],
                                                                        test_size=0.3,
                                                                        random_state=42)

            # Append the train and test sets to the lists
            X_train_list.append(X_train_q)
            X_test_list.append(X_test_q)
            y_train_list.append(y_train_q)
            y_test_list.append(y_test_q)

        # Concatenate all the train and test sets
        X_train = pd.concat(X_train_list)
        X_test = pd.concat(X_test_list)
        y_train = pd.concat(y_train_list)
        y_test = pd.concat(y_test_list)

        # Drop the 'quantile' column from the final datasets
        X_train.drop('quantile', axis=1, inplace=True)
        X_test.drop('quantile', axis=1, inplace=True)

    # Center and scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, [parameter_type, parameter_value]


if __name__ == '__main__':
    # Get paths
    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["ml_results"]
    csv_export_path = os.path.join(folder_paths["hyperparameters"], 'xgb_marching_cubess0_9.csv')
    csv_import_path = os.path.join(folder_paths["reconstructions_by_parameters"], 'marching_cubess0_9.csv')
    df = pd.read_csv(csv_import_path)

    print(df['parameter_type'].unique())

    # First run of train-test split to set the desired columns
    X_train, X_test, y_train, y_test = load_train_test_sets(df, by_year=False)

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
    results_xbg = pd.DataFrame()
    try:
        print('starting grid search')
        # Define distributions for hyperparameters
        param_dist = {
            'n_estimators': randint(50, 200),  # Number of boosting rounds
            'max_depth': randint(3, 10),  # Maximum depth of the trees
            'learning_rate': uniform(0.01, 0.3),  # Learning rate
            'min_child_weight': randint(1, 6)  # Minimum sum of instance weight needed in a child
        }

        # Perform random search with cross-validation
        random_search = RandomizedSearchCV(XGBRegressor(), param_distributions=param_dist, n_iter=100, cv=5,
                                           scoring='neg_mean_squared_error')
        random_search.fit(X_train, y_train)

        # Get the best hyperparameters found by random search
        best_params = random_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Refactor model training to use the best hyperparameters
        model = XGBRegressor(**best_params)
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

        # Plot predicted vs real values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, pred_val, color='blue', label='Predicted vs Real')
        plt.plot(y_test, y_test, color='red', linewidth=2, label='Ideal Line')
        plt.title('Predicted vs Real Values')
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

        # parameter_value = df['parameter_value'].unique()[0]
        # parameter_type = df['parameter_type'].unique()[0]
        #
        # current_results['Parameter_value'] = parameter_value
        # current_results['Parameter_name'] = parameter_type
        current_results['Regression_model'] = 'XGBoost'
        current_results['RMSE_score_calibration'] = mse_cal
        current_results['RMSE_score_validation'] = mse_val
        current_results['R2_score_calibration'] = r2_cal
        current_results['R2_score_validation'] = r2_val
        current_results['Successful_reconstructions_test'] = len(X_test)
        current_results['Successful_reconstructions_train'] = len(X_train)
        results_xbg = pd.concat([results_xbg, pd.DataFrame([current_results])], ignore_index=True)
    except ValueError:
        print('A small dataset. Cannot calculate')
    print(results_xbg.to_string())
    results_xbg.to_csv(csv_export_path, index=False)