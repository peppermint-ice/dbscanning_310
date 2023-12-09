import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


def fit_and_plot_model(x, y, model_name):
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
        X = sm.add_constant(np.column_stack((np.exp(x), x)))
    else:
        raise ValueError("Invalid model_name")

    model = sm.OLS(y, X).fit()
    r_squared = model.rsquared
    # print(model.params)
    # print('R2: ', model.rsquared)

    return model, model_name

    # # Plotting the results
    # plt.scatter(x, y, label='Data')
    # plt.plot(x, model.predict(X), label=f'{model_name.capitalize()} Model', linestyle='--')
    # plt.title(f'{model_name.capitalize()} Model\nR-squared: {r_squared:.4f}')
    # plt.xlabel(x.name)
    # plt.ylabel(y.name)
    # plt.legend()
    # plt.show()


csv_file_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes\df.csv'
csv2_file_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes\df2.csv'
df = pd.read_csv(csv_file_path)
print(df.to_string())

Y = df['Measured_leaf_area']
X = df.drop(columns=['File_name', 'Measured_leaf_area'], inplace=False)
X['Components_number'] = X['Components_number'].astype(float)
X['Flatness'] = X['Flatness'].abs()
print(X.to_string())
column_names = []
for columns in X.columns:
    column_names.append(columns)

models_list = []
model_names_list = []
df2 = pd.DataFrame(columns=column_names)


for model_type in ['linear', 'quadratic', 'cubic', 'logarithmic']:
    row = {}
    for parameter in X.columns:
        # print(model_type, parameter)
        model, model_name = fit_and_plot_model(X[parameter], Y, model_type)
        models_list.append(model)
        row[parameter] = model.rsquared
    row_df = pd.DataFrame([row], columns=row.keys())
    print(row)
    df2 = pd.concat([df, row_df], ignore_index=True)
print(df.to_string())
df2.to_csv(csv2_file_path, index=False)

X = df[['Height', 'Length', 'Width', 'Volume', 'Surface_area', 'Aspect_ratio', 'Elongation', 'Flatness', 'Sphericity', 'Compactness', 'Components_number']]


# Add a constant term to the model
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the model summary
print(model.summary())

# lasso = Lasso()
# param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
# grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X, Y)
# best_lasso = grid_search.best_estimator_
# lasso_model = best_lasso.fit(X, Y)


