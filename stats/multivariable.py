import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from config import paths
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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
    print('Start running')
    keys = [
        'Parameter_name',
        'Parameter_value',
        'Combination',
        'RMSE_score'
    ]

    multivars = dict.fromkeys(keys)
    results = pd.DataFrame()

    # Get path
    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["hyperparameters"]
    csv_import_path = os.path.join(folder_paths["data"], '001final.csv')
    df = pd.read_csv(csv_import_path)
    X_train, X_test, y_train, y_test = load_train_test_sets(df, by_year=True)
    print(len(X_train.columns))
    best_score = float('inf')
    best_features = None

    # Iterate through all possible combinations of features
    for parameter_type in df['parameter_type'].unique():
        df_param = df[df['parameter_type'] == parameter_type]
        for parameter_value in df_param['parameter_value'].unique():
            df_current = df_param[df_param['parameter_value'] == parameter_value]
            print(parameter_type)
            print(parameter_value)
            for r in range(1, len(X_train.columns)):
                for combination in itertools.combinations(X_train.columns, r):
                    # print(combination)
                    X_train, X_test, y_train, y_test = load_train_test_sets(df_current, by_year=True)
                    try:
                        # Fit a model using the selected features
                        model = LinearRegression()
                        model.fit(X_train[list(combination)], y_train)
                        # Evaluate the model on the testing set
                        y_pred = model.predict(X_test[list(combination)])
                        score = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        # Update the best score and features if the current combination performs better
                        if score < best_score:
                            best_score = score
                            best_features = combination
                        multivars['Parameter_value'] = parameter_value
                        multivars['Parameter_name'] = parameter_type
                        multivars['Combination'] = combination
                        multivars['RMSE_score'] = score
                        multivars['R2_score'] = r2
                        results = pd.concat([results, pd.DataFrame([multivars])], ignore_index=True)
                    except ValueError:
                        print('Too little data. No good')
            print('Columns: ')
            print(results.shape)
    results.to_csv(os.path.join(csv_folder_path, 'multivars2.csv'), index=False)
    print("Best combination of features:", best_features)
    print("Best MSE score:", best_score)