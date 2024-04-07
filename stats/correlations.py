import pandas as pd
import numpy as np
import os
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from matplotlib import pyplot as plt
from config import paths


def linear(data):
    linears = {
        'Parameter_name': [],
        'Parameter_value': [],
        'Correlating_parameter': [],
        'Model': [],
        'R2_score': [],
    }
    for parameter_type in data['parameter_type'].unique():
        df_param = data[data['parameter_type'] == parameter_type]
        for parameter_value in df_param['parameter_value'].unique():
            df_current = df_param[df_param['parameter_value'] == parameter_value]
            df_y = df_current['Measured_leaf_area']
            X = df_current.drop(columns=['File_name', 'Measured_leaf_area', 'parameter_type', 'parameter_value'], inplace=False)
            X['Components_number'] = X['Components_number'].astype(float)
            X['Rectangular_area'] = X['Length'] * X['Width']
            X['Sum_of_dimensions'] = X['Length'] + X['Width']
            X['Flatness'] = X['Flatness'].abs()
            column_names = []
            for columns in X.columns:
                column_names.append(columns)
            X = sm.add_constant(X)
            for x_param in X.columns:
                model = sm.OLS(df_y, X[x_param]).fit()
                r_squared = model.rsquared
                linears['Parameter_value'].append(parameter_value)
                linears['Parameter_name'].append(parameter_type)
                linears['Correlating_parameter'].append(x_param)
                linears['Model'].append(model)
                linears['R2_score'].append(r_squared)
    return linears


def multivariable(folder_path):
    return


def rforest(folder_path):
    return


def svm(folder_path):
    return

def xgboost(folder_path):
    return


def plot_max_r_squared(linears):
    # Group data by Parameter_name
    grouped_data = pd.DataFrame(linears).groupby('Parameter_name')

    for parameter_name, group in grouped_data:
        max_r_squared_values = group.groupby('Parameter_value')['R2_score'].max()
        parameter_values = max_r_squared_values.index
        r_squared_scores = max_r_squared_values.values

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.8 / len(parameter_values)  # Adjust the width of the bars

        # Calculate position for each bar
        positions = np.arange(len(parameter_values))

        # Initialize color map and legend entries
        color_map = {}
        legend_entries = {}

        for position, value, r_squared in zip(positions, parameter_values, r_squared_scores):
            # Find correlating parameter with best R-squared
            best_corr_param_idx = group[group['Parameter_value'] == value]['R2_score'].idxmax()
            best_corr_param = linears['Correlating_parameter'][best_corr_param_idx]

            # Check if color has already been assigned to this parameter
            if best_corr_param not in color_map:
                # Assign a unique color to the correlating parameter
                color_map[best_corr_param] = plt.cm.get_cmap('viridis', len(color_map))

            color = color_map[best_corr_param](0.5)  # Use mid-tone color from colormap

            # Add legend entry if not already present
            if best_corr_param not in legend_entries:
                legend_entries[best_corr_param] = ax.bar(position, r_squared, color=color, label=f'{value}', alpha=0.7)
            else:
                ax.bar(position, r_squared, color=color, alpha=0.7)

        ax.set_title(f'Maximum R-squared for {parameter_name}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Maximum R-squared')
        ax.grid(True)
        ax.set_xticks(positions)
        ax.set_xticklabels(parameter_values, rotation=45)

        # Create legend with unique color assignments
        handles = []
        labels = []
        for param, cmap in color_map.items():
            handles.append(plt.Rectangle((0, 0), 1, 1, color=cmap(0.5)))
            labels.append(param)
        ax.legend(handles, labels, title='Correlating Parameter', loc='upper left')

        plt.tight_layout()
        plt.show()


def plot_prediction(linears):
    # Group data by Parameter_name
    grouped_data = pd.DataFrame(linears).groupby('Parameter_name')

    for parameter_name, group in grouped_data:
        # Find the index of the model with the highest R-squared value
        best_model_idx = group['R2_score'].idxmax()
        best_model = linears['Model'][best_model_idx]

        # Extract predictor variables
        X = group.drop(columns=['File_name', 'Measured_leaf_area', 'parameter_type', 'parameter_value'], inplace=False)
        X['Components_number'] = X['Components_number'].astype(float)
        X['Rectangular_area'] = X['Length'] * X['Width']
        X['Sum_of_dimensions'] = X['Length'] + X['Width']
        X['Flatness'] = X['Flatness'].abs()
        X = sm.add_constant(X)

        # Make predictions using the best model
        predicted_values = best_model.predict(X)

        # Plot actual vs predicted values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(group['Measured_leaf_area'], predicted_values, color='blue', alpha=0.7)
        ax.plot(group['Measured_leaf_area'], group['Measured_leaf_area'], color='red', linestyle='--')  # Diagonal line
        ax.set_title(f'Prediction vs Actual for {parameter_name}')
        ax.set_xlabel('Actual Measured Leaf Area')
        ax.set_ylabel('Predicted Leaf Area')
        ax.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print('Start running')

    # Get path
    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["hyperparameters"]
    csv_import_path = os.path.join(folder_paths["data"], '001final.csv')
    df = pd.read_csv(csv_import_path)
    linears = linear(df)
    for i, r_squared in enumerate(linears['R2_score']):
        parameter_name = linears['Parameter_name'][i]
        parameter_value = linears['Parameter_value'][i]
        corresponding_param = linears['Correlating_parameter'][i]
        print(f"R-squared: {r_squared}, Parameter Name: {parameter_name}, Parameter Value: {parameter_value}, Corresponding Parameter: {corresponding_param}")

    # Get path
    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["hyperparameters"]
    csv_import_path = os.path.join(folder_paths["data"], '001final.csv')
    df = pd.read_csv(csv_import_path)
    linears = linear(df)
    plot_max_r_squared(linears)
    plot_prediction(linears)