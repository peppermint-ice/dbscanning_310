import pandas as pd
import os
import sys
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

from config import paths


def load_train_test_sets(df, target_column='Measured_leaf_area', by_year=False):
    '''
    A function to load train and test data sets based on a target column and year.
    :param df: dataframe containing reconstruction results for one parameter name and value
    :param target_column: i guess, leaf area. might be changed to whatever is needed.
    :param by_year: if true, allows to train models on year 2023 and test on year 2024.
    :return: 4 dataframes containing train and test data sets.
    '''
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
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1), df[target_column],
                                                            test_size=0.3, random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # print(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Get the file path from the command-line argument
    file_path = sys.argv[1]

    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["multilinear_results"]

    # Read the CSV file specified by the command-line argument
    df = pd.read_csv(file_path)

    print(df['parameter_type'].unique())

    parameter_value = df['parameter_value'].unique()[0]
    parameter_type = df['parameter_type'].unique()[0]

    # First run of train-test split to set the desired columns
    X_train, X_test, y_train, y_test = load_train_test_sets(df, by_year=True)

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
    for r in range(1, len(X_train.columns)):
        for combination in itertools.combinations(X_train.columns, r):
            try:
                # Refactor model training to use the best hyperparameters
                model = LinearRegression()
                model.fit(X_train[list(combination)], y_train)
                pred_cal = model.predict(X_train[list(combination)])
                pred_val = model.predict(X_test[list(combination)])
                mse_cal = mean_squared_error(y_train, pred_cal)
                mse_val = mean_squared_error(y_test, pred_val)
                r2_cal = r2_score(y_train, pred_cal)
                r2_val = r2_score(y_test, pred_val)

                # print(parameter_type, ': ', parameter_value)
                # print('Correlating parameter: ', column)
                # print('r2 cal: ', r2_cal)
                # print('r2 val: ', r2_val)
                # print('')

                current_results['Parameter_value'] = parameter_value
                current_results['Parameter_name'] = parameter_type
                current_results['Regression_model'] = 'Multilinear Regression'
                current_results['Parameters'] = combination
                current_results['RMSE_score_calibration'] = mse_cal
                current_results['RMSE_score_validation'] = mse_val
                current_results['R2_score_calibration'] = r2_cal
                current_results['R2_score_validation'] = r2_val
                current_results['Successful_reconstructions_test'] = len(X_test)
                current_results['Successful_reconstructions_train'] = len(X_train)
                results_rf = pd.concat([results_rf, pd.DataFrame([current_results])], ignore_index=True)
                print(results_rf.shape)
            except ValueError:
                print('A small dataset. Cannot calculate')
    output_file = str(parameter_value) + parameter_type + '_results_multilinear.csv'
    output_file_path = os.path.join(csv_folder_path, output_file)
    results_rf.to_csv(output_file_path, index=False)
