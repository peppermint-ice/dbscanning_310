import pandas as pd
from matplotlib import pyplot as plt
import os
import re


def create_all_data_csv(folder_path, file_name):
    # Create folder & file names
    plys = os.listdir(folder_path)
    csv_path = os.path.join(folder_path, file_name)
    print(csv_path)

    # Create a dictionary
    plant_data = {
        'Plant Number': [],
        'Cut Number': [],
        'Repetition': [],
        'Date': [],
        'Measured Leaf Area': [],
        'Estimated Leaf Area': [],
        'Predicted Leaf Area': []
    }

    # Create a regex to extract data from file names
    pattern = r"(\d+)_(\d+)_(\d+).?(\d*)_(\d+)p(\d+).ply"

    # Append the dictionary to create a future df with all the data
    for file in plys:
        ply_file_path = os.path.join(folder_path, file)
        if os.path.isfile(ply_file_path) and re.search('.ply', file):
            match = re.search(pattern, file)
            print(file)
            plant_data['Plant Number'].append(match.group(2))
            plant_data['Cut Number'].append(match.group(3))
            plant_data['Repetition'].append(match.group(4))
            plant_data['Date'].append(str(match.group(1)))
            plant_data['Measured Leaf Area'].append(int(match.group(5)) + int(match.group(6)) / 10 ** len(match.group(6)))
            plant_data['Estimated Leaf Area'].append('')
            plant_data['Predicted Leaf Area'].append('')

    # Create a df
    df = pd.DataFrame(plant_data)

    # Sort the df to have everything in correct order
    df['Plant Number'] = df['Plant Number'].astype(int)
    df['Cut Number'] = df['Cut Number'].astype(int)
    df['Date'] = df['Date'].astype(int)
    df = df.sort_values(by=['Date', 'Plant Number', 'Cut Number', 'Repetition'], ignore_index=True)

    # Print and save the data
    print(df.to_string())
    df.to_csv(csv_path)
    return df


def print_hists(df, plot_folder):
    # Create a plot
    plt.hist(df['Measured Leaf Area'], rwidth=0.7)
    plt.xlabel('Measured Leaf Area, cm2')
    plt.ylabel('Number of occurrences')
    filename = os.path.join(plot_folder, 'LA_distribution.png')

    # Create the text with info
    text = ['Number of plants: ' + str(df.groupby('Date')['Plant Number'].nunique().sum()),
            'Number of measurements: ' + str(len(df['Plant Number'])),
            'Min leaf area: ' + str(min(df['Measured Leaf Area'])) + 'cm2',
            'Max leaf area: ' + str(max(df['Measured Leaf Area'])) + 'cm2']
    position = 80
    for line in text:
        plt.text(3000, position, line, fontsize=10)
        position -= 5
    # Show and save the figure
    # plt.show()
    plt.savefig(filename, transparent=True)


if __name__ == "__main__":
    # Create all paths
    ply_folder_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes'
    output_folder = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes\plots\raw_data'
    csv_file_name = 'complete_la_data.csv'

    # Create the csv
    df = create_all_data_csv(ply_folder_path, csv_file_name)

    # Create the plot
    print_hists(df, output_folder)
