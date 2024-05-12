import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from config import paths

def preprocess_data(df, target_column='Measured_leaf_area', by_year=False):
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

    return X_train_scaled, X_test_scaled, y_train, y_test


def preprocess_data_old(df, target_column='Measured_leaf_area', by_year=False):
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


def generate_test_data(num_rows=1000):
    np.random.seed(42)
    data = {
        'A': np.random.randint(1, 100, size=num_rows),
        'B': np.random.randint(1000, 10000, size=num_rows),
        'File_name': ['file_' + str(i) for i in range(num_rows)],
        'Measured_leaf_area': np.random.randint(1, 1000, size=num_rows),
        'parameter_type': np.random.choice(['A', 'B', 'C'], size=num_rows),
        'parameter_value': np.random.rand(num_rows)
    }
    df = pd.DataFrame(data)
    return df



def plot_distributions(y_train, y_test):
    plt.figure(figsize=(8, 6))
    plt.hist(y_train, bins=20, color='blue', alpha=0.5, label='y_train', density=True)
    plt.hist(y_test, bins=20, color='green', alpha=0.5, label='y_test', density=True)
    plt.title('Distribution of Measured Leaf Area')
    plt.xlabel('Measured Leaf Area')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

folder_paths = paths.get_paths()
folder_path = folder_paths["reconstructions_by_parameters"]
data = pd.read_csv(folder_path + "/marching_cubess0_9.csv")

print(data.shape)
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data_old(data, target_column='Measured_leaf_area', by_year=False)
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

plot_distributions(y_train, y_test)