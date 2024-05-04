import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from config import paths


def preprocess_data(df, target_column='Measured_leaf_area', by_year=False):
    parameter_type = df['parameter_type'].unique()[0]
    parameter_value = df['parameter_value'].unique()[0]
    df.loc[:, 'Year'] = ('20' + df['File_name'].str[:6].str[-2:]).astype(int)
    df_train = df[df['Year'] == 2023]
    df_test = df[df['Year'] == 2024]
    if by_year:
        X_train = df_train.drop(columns=[target_column, 'File_name', 'parameter_type', 'parameter_value', 'Year', 'Surface_area', 'Elongation', 'Flatness', 'Sphericity', 'Compactness'],
                                inplace=False, axis=1)
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column, 'File_name', 'parameter_type', 'parameter_value', 'Year', 'Surface_area', 'Elongation', 'Flatness', 'Sphericity', 'Compactness'],
                              inplace=False, axis=1)
        y_test = df_test[target_column]
    else:
        df.drop(columns=['File_name', 'parameter_type', 'parameter_value', 'Year', 'Surface_area', 'Elongation', 'Flatness', 'Sphericity', 'Compactness'],
                inplace=True, axis=1)
        # Define the number of quantiles
        num_quantiles = int(10)

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
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    return X_train, X_test, y_train, y_test, [parameter_type, parameter_value]


def bootstrap_sample(X, y, num_samples=100):
    sample_indices = np.random.choice(range(len(X)), size=(num_samples, len(X.columns)), replace=True)
    X_samples = [X.iloc[idx] for idx in sample_indices]
    y_samples = [y.iloc[idx] for idx in sample_indices]
    return X_samples, y_samples


def plot_predictions(y_true, y_pred, r2, parameter_value, parameter_type):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    plt.xlabel('Measured Leaf Area')
    plt.ylabel('Predicted Leaf Area')
    plt.title('Leaf Area Prediction: RF\n{parameter_type}, {parameter_value}\nR2 Score: {:.2f}'.format(r2))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{parameter_value}_{parameter_type}_prediction_plot_rf.png')



if __name__ == '__main__':
    # Get the file path from the command-line argument
    file_path = sys.argv[1]

    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["ml_results"]

    # Read the CSV file specified by the command-line argument
    df = pd.read_csv(file_path)

    print(df['parameter_type'].unique())

    # First run of train-test split to set the desired columns
    X_train, X_test, y_train, y_test, values = preprocess_data(df, by_year=False)

    keys = [
        'Parameter_name',
        'Parameter_value',
        'Regression_model',
        'RMSE_score_calibration',
        'RMSE_score_validation',
        'R2_score_calibration',
        'R2_score_validation',
        'Successful_reconstructions_test',
        'Successful_reconstructions_train']
    current_results = dict.fromkeys(keys)
    results_rf = pd.DataFrame()
    try:
        print('starting grid search')
        # Define distributions for hyperparameters
        param_dist = {
            'n_estimators': randint(50, 200),  # Number of trees in the forest
            'max_depth': [None] + list(randint(3, 10).rvs(5)),  # Maximum depth of the trees
            'min_samples_split': randint(2, 20),  # Minimum number of samples required to split a node
            'min_samples_leaf': randint(1, 10)  # Minimum number of samples required at each leaf node
        }

        # Perform random search with cross-validation
        random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist, n_iter=100, cv=5,
                                           scoring='neg_mean_squared_error')
        random_search.fit(X_train, y_train)

        # Get the best hyperparameters found by random search
        best_params = random_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Refactor model training to use the best hyperparameters
        model = RandomForestRegressor(**best_params)
        # Bootstrap sampling
        X_train_samples, y_train_samples = bootstrap_sample(X_train, y_train)

        mse_cal = []
        mse_val = []
        r2_cal = []
        r2_val = []

        for X_train_sample, y_train_sample in zip(X_train_samples, y_train_samples):
            model.fit(X_train_sample, y_train_sample)
            pred_cal = model.predict(X_train_sample)
            pred_val = model.predict(X_test)
            mse_cal.append(mean_squared_error(y_train_sample, pred_cal))
            mse_val.append(mean_squared_error(y_test, pred_val))
            r2_cal.append(r2_score(y_train_sample, pred_cal))
            r2_val.append(r2_score(y_test, pred_val))

        mse_cal_mean = np.mean(mse_cal)
        mse_val_mean = np.mean(mse_val)
        r2_cal_mean = np.mean(r2_cal)
        r2_val_mean = np.mean(r2_val)

        parameter_value = values[1]
        parameter_type = values[0]

        current_results['Parameter_value'] = parameter_value
        current_results['Parameter_name'] = parameter_type
        current_results['Regression_model'] = 'RandomForest'
        current_results['RMSE_score_calibration'] = mse_cal_mean
        current_results['RMSE_score_validation'] = mse_val_mean
        current_results['R2_score_calibration'] = r2_cal_mean
        current_results['R2_score_validation'] = r2_val_mean
        current_results['Successful_reconstructions_test'] = len(X_test)
        current_results['Successful_reconstructions_train'] = len(X_train)
        results_rf = pd.concat([results_rf, pd.DataFrame([current_results])], ignore_index=True)
        print(results_rf.shape)
        output_file = f'{parameter_value}_{parameter_type}_results_rf2.csv'
        output_file_path = os.path.join(csv_folder_path, output_file)
        results_rf.to_csv(output_file_path, index=False)

        # Plot predictions
        plot_predictions(y_test, model.predict(X_test), r2_val_mean, parameter_value, parameter_type)

    except ValueError:
        print('A small dataset. Cannot calculate')