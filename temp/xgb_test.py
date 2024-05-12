import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, stats
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap
from joblib import dump
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Load the data from the given file path.

    Parameters:
        file_path (str): The file path of the data file.

    Returns:
        DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)


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


def bootstrap_split_with_ids(X_train, y_train, train_size_share, iterations=100, random_state=15):
    """
    Generate bootstrap samples.

    Parameters:
        trn (ndarray): The input data.
        train_size (int): Size of the training set.
        test_size (int): Size of the test set.
        iterations (int): Number of bootstrap iterations.
        random_state (int): Random state for reproducibility.

    Yields:
        tuple: Bootstrap train and test sets.
    """

    trn = np.column_stack((X_train, y_train))
    train_size = int(train_size_share * len(trn))
    test_size = len(trn) - train_size

    num_samples = len(trn)
    data_with_ids = np.column_stack((trn, np.arange(num_samples)))  # Add an identifier to each data point

    # Create a random number generator with a fixed state
    rng = np.random.RandomState(random_state)

    for _ in range(iterations):
        # Generate random indices with replacement for the training set using the fixed random state
        train_indices = rng.choice(num_samples, size=train_size, replace=True)

        # Get unique IDs from the training set
        train_ids = set(np.unique(data_with_ids[train_indices, -1]))

        # Efficiently prepare pool of indices for test set selection, excluding training IDs
        possible_test_indices = list(set(range(num_samples)) - train_ids)

        # Generate random indices with replacement for the test set from the remaining indices
        test_indices = rng.choice(possible_test_indices, size=test_size, replace=True)

        train_set = data_with_ids[train_indices]
        test_set = data_with_ids[test_indices]

        yield train_set, test_set

def random_search_xgb(X_train, y_train):
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
    return best_params


def apply_xgb_for_iteration(trn, train_size):
    num_samples = data.shape[0]

        # Counter to track occurrences of each data point in train and test
    train_occurrences = {i: 0 for i in range(num_samples)}
    test_occurrences = {i: 0 for i in range(num_samples)}

        #create 100 boottrap ample
    bootstrap_i = enumerate(bootstrap_split_with_ids(trn, train_size, test_size, iterations=100))
    # Iterate through bootstrap samples
    for i, (train_set, test_set) in bootstrap_i:
        X_train = train_set[:, :-2]  # Exclude identifier and Y
        Y_train = train_set[:, -2]   # Identifier
        X_test = test_set[:, :-2]

        xgb.fit(X_train, Y_train)
        train_ids = []
        test_ids = []

                # Increment occurrences for each data point in train and test
        for idx in train_set[:, -1]:
            train_occurrences[int(idx)] += 1
            train_ids.append(idx)

        for idx in test_set[:, -1]:
            test_occurrences[int(idx)] += 1
            test_ids.append(idx)

        Y_train_pred = xgb.predict(X_train)
        Y_test_pred = xgb.predict(X_test)

        # Save the model to disk
        model_filename = f'/content/drive/MyDrive/Colab Notebooks/ML_models_LAI/LI_model/spxgb_model_{i}.joblib'  # Change the path as needed
        dump(xgb, model_filename)

        yield train_ids, test_ids, Y_train, Y_train_pred, Y_test_pred, xgb


def train_xgb_model(X_train, Y_train, param_distributions, cv):
    """
    Train an XGBoost model.

    Parameters:
        X_train (ndarray): Training data.
        Y_train (ndarray): Target for training data.
        param_distributions (dict): Hyperparameter distributions for RandomizedSearchCV.
        cv (KFold): Cross-validation object.

    Returns:
        RandomizedSearchCV: Trained XGBoost model.
    """
    xgb_model = XGBRegressor()
    rnd_search = RandomizedSearchCV(estimator=xgb_model,
                                    param_distributions=param_distributions,
                                    n_iter=50,
                                    cv=cv,
                                    verbose=2,
                                    random_state=42,
                                    n_jobs=-1)
    rnd_search.fit(X_train, Y_train)
    return rnd_search


def save_model(model, file_path):
    """
    Save the trained model.

    Parameters:
        model: Trained model.
        file_path (str): File path for saving the model.
    """
    dump(model, file_path)


def calculate_metrics(true_values, predicted_values):
    """
    Calculate evaluation metrics.

    Parameters:
        true_values (ndarray): True target values.
        predicted_values (ndarray): Predicted target values.

    Returns:
        tuple: Evaluation metrics (R-squared, RMSE, NRMSE, RPD).
    """
    r_squared = np.square(np.corrcoef(true_values, predicted_values)[0, 1])
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    nrmse = rmse / (np.max(true_values) - np.min(true_values))
    rpd = np.std(true_values) / rmse
    return r_squared, rmse, nrmse, rpd


def save_metrics_to_csv(metrics, file_path):
    """
    Save evaluation metrics to a CSV file.

    Parameters:
        metrics (dict): Dictionary containing evaluation metrics.
        file_path (str): File path for saving the CSV.
    """
    pd.DataFrame(metrics, index=[0]).to_csv(file_path, index=False)


def plot_scatter(true_values, predicted_values, file_path):
    """
    Plot a scatter plot of true values vs predicted values.

    Parameters:
        true_values (ndarray): True target values.
        predicted_values (ndarray): Predicted target values.
        file_path (str): File path for saving the plot.
    """
    plt.figure(figsize=(6, 6))
    sns.regplot(x=true_values, y=predicted_values, scatter_kws={'s': 50}, label='Regression Line', ci=95)
    plt.plot([0, np.max(true_values)], [0, np.max(true_values)], linestyle='dashed', color='black', label='1:1 Line')
    plt.xlabel(r'LAI$_m$ (m$^2$ m$^{-2}$)')
    plt.ylabel(r'LAI$_p$ (m$^2$ m$^{-2}$)')
    plt.title('Scatterplot of True Value vs. Mean Prediction')
    plt.legend()
    plt.savefig(file_path)
    plt.close()


def calculate_shap_values(xgb_models, X, background, names, file_path):
    """
    Calculate and save SHAP values.

    Parameters:
        xgb_models (list): List of trained XGBoost models.
        X (ndarray): Input data.
        background (ndarray): Background data for SHAP values calculation.
        names (list): Feature names.
        file_path (str): File path for saving SHAP values.
    """
    all_shap_values = []

    for xgb_model in xgb_models:
        explainer = shap.TreeExplainer(model=xgb_model, data=background)
        shap_values = explainer.shap_values(X)
        all_shap_values.append(shap_values)

    all_shap_values = np.array(all_shap_values)
    mean_shap_values = np.mean(all_shap_values, axis=0)

    mean_shap_df = pd.DataFrame(mean_shap_values.squeeze(), columns=names)
    mean_shap_df.to_csv(file_path, index=False)


def main():
    # Load data
    data = load_data('/content/drive/MyDrive/Colab Notebooks/WheatDryFace/WDF_23_LiDAR/ResampSpear_LiMet.csv')

    # Preprocess data
    X_train, Y_train, X_test, Y_test = preprocess_data(data)

    # Create a list of hyperparameter values to search over
    param_distributions = {
        'max_depth': [3, 5, 7],
        'min_child_weight': [3, 5, 7],
        'subsample': [0.1, 0.5, 1.0],
        'colsample_bytree': [0.1, 0.5, 1.0],
        'learning_rate': [0.01, 0.1, 0.3],
        'gamma': [0.5, 1, 1.5],
        'n_estimators': [10, 50, 100, 500, 1000, 1500]
    }

    # K-Fold cross-validator
    cv = KFold(n_splits=5)

    # Train XGBoost model
    rnd_search = train_xgb_model(X_train, Y_train, param_distributions, cv)

    # Save the best model
    save_model(rnd_search.best_estimator_,
               '/content/drive/MyDrive/Colab Notebooks/ML_models_LAI/LI_model/spxgb_model.joblib')

    # Calculate metrics for train set
    train_predictions = rnd_search.best_estimator_.predict(X_train)
    train_metrics = calculate_metrics(Y_train, train_predictions)

    # Save train set metrics to CSV
    train_metrics_file = '/content/drive/MyDrive/Colab Notebooks/WheatDryFace/WDF_23_LiDAR/Predicted_Model_dfs/Trn_xgb_metrics.csv'
    save_metrics_to_csv({'R-squared': train_metrics[0],
                         'RMSE': train_metrics[1],
                         'NRMSE': train_metrics[2],
                         'RPD': train_metrics[3]}, train_metrics_file)

    # Calculate metrics for test set
    test_predictions = rnd_search.best_estimator_.predict(X_test)
    test_metrics = calculate_metrics(Y_test, test_predictions)

    # Save test set metrics to CSV
    test_metrics_file = '/content/drive/MyDrive/Colab Notebooks/WheatDryFace/WDF_23_LiDAR/Predicted_Model_dfs/Test_xgb_metrics.csv'
    save_metrics_to_csv({'R-squared': test_metrics[0],
                         'RMSE': test_metrics[1],
                         'NRMSE': test_metrics[2],
                         'RPD': test_metrics[3]}, test_metrics_file)

    # Plot scatter plot for train set
    train_plot_file = '/content/drive/MyDrive/Colab Notebooks/WheatDryFace/WDF_23_LiDAR/Predicted_Model_dfs/Train_Scatterplot.png'
    plot_scatter(Y_train, train_predictions, train_plot_file)

    # Plot scatter plot for test set
    test_plot_file = '/content/drive/MyDrive/Colab Notebooks/WheatDryFace/WDF_23_LiDAR/Predicted_Model_dfs/Test_Scatterplot.png'
    plot_scatter(Y_test, test_predictions, test_plot_file)

    # Calculate SHAP values
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    names = data.columns[14:-1]
    calculate_shap_values([rnd_search.best_estimator_], X_test, background, names,
                          '/content/drive/MyDrive/Colab Notebooks/WheatDryFace/WDF_23_LiDAR/mean_shap_values_xgb_li_sp.csv')


if __name__ == "__main__":
    main()
