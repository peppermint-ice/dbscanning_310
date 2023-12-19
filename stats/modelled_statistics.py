import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os


def create_all_regressions_plots(df, output_folder, parameters):
    # Specify output file name:
    filename = os.path.join(output_folder, 'all_regressions.png')

    # Set up the subplot matrix
    fig, axes = plt.subplots(len(parameters), len(parameters), figsize=(15, 15))

    # Loop through each combination of parameters and plot linear regression
    for i, param1 in enumerate(parameters):
        for j, param2 in enumerate(parameters):
            # Skip subplots where i < j (lower triangle)
            if i < j:
                continue

            # Scatter plot with x and y switched
            sns.scatterplot(x=param2, y=param1, data=df, ax=axes[i, j])

            # Fit linear regression model
            X = df[param2].values.reshape(-1, 1)
            y = df[param1].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            # Calculate R-squared
            r2 = r2_score(y, y_pred)

            # Plot regression line
            # Create a colormap
            cmap = cm.get_cmap('RdYlGn')  # You can choose any colormap you prefer

            # Map the normalized R-squared to a color
            color = cmap(r2)
            # Draw a regression line
            axes[i, j].plot(X, y_pred, color=color, linewidth=2)

            # Annotate with R-squared (set R-squared to 1 if param1 equals param2)
            r2 = 1 if param1 == param2 else r2
            axes[i, j].text(0.5, 0.95, f'R² = {r2:.3f}', transform=axes[i, j].transAxes, ha='center', va='top',
                            fontsize=8)

            # Remove x-axis ticks
            if i < len(parameters) - 1:
                axes[i, j].set_xticks([])
            if j > 0:
                axes[i, j].set_yticks([])
            # Remove plot labels
            axes[i, j].set(ylabel='')
            axes[i, j].set(xlabel='')

    # Set column labels
    for i, param in enumerate(parameters):
        axes[len(parameters) - 1, i].set_xlabel(param)

    # Set row labels
    for i, param in enumerate(parameters):
        axes[i, 0].set_ylabel(param, rotation=45, ha='right', rotation_mode='anchor')

    # Remove empty subplot boxes
    for i in range(len(parameters)):
        for j in range(len(parameters)):
            if i < j:
                axes[i, j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(filename, transparent=True)
    plt.show()


def plot_every_parameter_hist(df, output_folder, parameters):
    for parameter in parameters:
        # Specify file name
        filename = os.path.join(output_folder, parameter + '.png')
        plt.hist(df[parameter], rwidth=0.7)
        plt.xlabel(parameter)
        plt.savefig(filename, transparent=True)
        # plt.show()
        plt.clf()
    return


if __name__ == '__main__':
    # Specify paths to dfs
    leaf_area_csv_path = r'D:\results\plys\complete_la_data.csv'
    la_df = pd.read_csv(leaf_area_csv_path)

    alpha_shape_csv_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes\df.csv'
    as_df = pd.read_csv(alpha_shape_csv_path)

    rsq_csv_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes\df2.csv'
    rsq_df = pd.read_csv(rsq_csv_path)

    # Specify plot folder path
    output_folder = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes\plots\modelled_data'

    # Create big plot with all regressions
    # Specify the parameters
    parameters = ['Height', 'Length', 'Width', 'Volume', 'Surface_area', 'Aspect_ratio', 'Elongation', 'Flatness',
                  'Sphericity', 'Compactness', 'Components_number', 'Point_density']

    # Run the function
    create_all_regressions_plots(as_df, output_folder, parameters)

    # Run the function to build all histograms
    plot_every_parameter_hist(as_df, output_folder, parameters)
