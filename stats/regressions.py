import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import os
from matplotlib.cm import get_cmap


def fit_and_plot_model(x, y, model_name, output_folder):
    # Linear regression
    if model_name == 'linear':
        X = sm.add_constant(x)
    # Quadratic regression
    elif model_name == 'quadratic':
        X = sm.add_constant(np.column_stack((x, x**2)))
    # Cubic regression
    elif model_name == 'cubic':
        X = sm.add_constant(np.column_stack((x, x**2, x**3)))
    # Logarithmic regression
    elif model_name == 'logarithmic':
        X = sm.add_constant(np.column_stack((np.log(x), x)))
    # Exponential regression
    elif model_name == 'exponential':
        x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
        X = sm.add_constant(np.column_stack((np.exp(x_normalized), x_normalized)))
    else:
        raise ValueError("Invalid model_name")

    model = sm.OLS(y, X).fit()
    r_squared = model.rsquared
    # print(model.params)
    print('R2: ', model.rsquared)

    # Plotting the results
    if model_name == 'exponential':
        plt.scatter(x_normalized, y, label='Data')
    else:
        plt.scatter(x, y, label='Data')

    # Generate x values for prediction
    if model_name == 'exponential':
        x_fit = np.linspace(min(x_normalized), max(x_normalized), 100)
    else:
        x_fit = np.linspace(min(x), max(x), 100)

    # Prepare X_fit based on the model type
    if model_name == 'linear':
        X_fit = sm.add_constant(x_fit)
    elif model_name == 'quadratic':
        X_fit = sm.add_constant(np.column_stack((x_fit, x_fit**2)))
    elif model_name == 'cubic':
        X_fit = sm.add_constant(np.column_stack((x_fit, x_fit**2, x_fit**3)))
    elif model_name == 'logarithmic':
        X_fit = sm.add_constant(np.column_stack((np.log(x_fit), x_fit)))
    elif model_name == 'exponential':
        X_fit = sm.add_constant(np.column_stack((np.exp(x_fit), x_fit)))

    y_fit = model.predict(X_fit)

    plt.plot(x_fit, y_fit, label=f'{model_name.capitalize()} Model', linestyle='--')
    plt.title(f'{model_name.capitalize()} Model\nR-squared: {r_squared:.4f}')
    plt.xlabel(x.name.replace('_', ' '))
    plt.ylabel(y.name.replace('_', ' '))
    # plt.legend()

    # Save the figure
    filename = f"{x.name}_{model_name}.png"
    filepath = os.path.join(output_folder, filename)
    print(filepath)
    plt.savefig(filepath, transparent=True)

    # Show the plot
    # plt.show()
    plt.clf()
    return model, model_name


csv_file_path = r'G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\LA\plys\data\alphas.csv'
csv2_file_path = r'G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\LA\plys\data\alphas_reg.csv'
output_folder = r'G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\LA\plys\data\plots\alphas\regressions'
df = pd.read_csv(csv_file_path)
print(df.to_string())

Y = df['Measured_leaf_area']
X = df.drop(columns=['File_name', 'Measured_leaf_area'], inplace=False)
X['Components_number'] = X['Components_number'].astype(float)
X['Rectangular_area'] = X['Length'] * X['Width']
X['Sum_of_dimensions'] = X['Length'] + X['Width']
X['Flatness'] = X['Flatness'].abs()
print(X.to_string())
column_names = []
for columns in X.columns:
    column_names.append(columns)

models_list = []
model_names_list = []
df2 = pd.DataFrame(columns=column_names)


for model_type in ['linear', 'quadratic', 'cubic', 'logarithmic', 'exponential']:
    row = {}
    for parameter in X.columns:
        print(model_type, parameter)
        model, model_name = fit_and_plot_model(X[parameter], Y, model_type, output_folder)
        models_list.append(model)
        row[parameter] = model.rsquared
    row_df = pd.DataFrame([row], columns=row.keys())
    print(row)
    df2 = pd.concat([df2, row_df], ignore_index=True)
print(df2.to_string())
df2.to_csv(csv2_file_path, index=False)

X = df[['Height', 'Length', 'Width', 'Volume', 'Surface_area', 'Aspect_ratio', 'Elongation', 'Flatness', 'Sphericity', 'Compactness', 'Components_number']]
X['Rectangular_area'] = X['Length'] * X['Width']
X = X[['Surface_area', 'Volume', 'Components_number', 'Height']]

# check linear regression
# Add a constant term to the model
X = sm.add_constant(X)
# Fit the linear regression model
model = sm.OLS(Y, X).fit()
# Print the model summary
print(model.summary())

# #check quadratic regression
# X = df[['Height', 'Length', 'Width', 'Volume', 'Surface_area', 'Aspect_ratio', 'Elongation', 'Flatness', 'Sphericity', 'Compactness', 'Components_number']]
# X = df[['Surface_area', 'Volume', 'Components_number']]
# X = sm.add_constant(np.column_stack((X, X**2)))
# # Fit the quadratic regression model
# model = sm.OLS(Y, X).fit()
# # Print the quadratic model summary
# print(model.summary())

# Plotting the parameters
# for column in df.columns:
#     plt.hist(df[column])
#     plt.title(column)
    # plt.show()


# Plotting the R2s
rsquared_df = pd.read_csv(csv2_file_path)

rsquared_df.index = ['linear', 'quadratic', 'cubic', 'logarithmic', 'exponential']

# Set the color map for each model type
cmap = get_cmap('viridis', len(rsquared_df.index))

# Set up the bar chart
bar_width = 0.2
bar_positions = np.arange(len(rsquared_df.columns))
models = rsquared_df.index

# Plot the bars for each model type
for i, model in enumerate(models):
    color = cmap(i)
    plt.bar(bar_positions + i * bar_width, rsquared_df.loc[model], width=bar_width, label=model.capitalize(), color=color)

# Set labels and title
plt.ylabel('R2')
plt.title('R2 for Different Models')
plt.xticks(bar_positions + bar_width * (len(models) / 2), rsquared_df.columns.str.replace('_', ' '), rotation=45, ha='right')  # Adjust x-axis labels

# Add legend
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'total.png', dpi=300), transparent=True)
# Show the plot
# plt.show()


