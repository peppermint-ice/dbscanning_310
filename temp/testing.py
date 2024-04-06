from config import paths
import os
import re
import pandas as pd

# Get path
folder_paths = paths.get_paths()

# Set folders
corrected_folder_path = folder_paths["corrected"]
csv_folder_path = folder_paths["hyperparameters"]

csvs = os.listdir(csv_folder_path)

df = pd.DataFrame()

for file in csvs:
    file_path = os.path.join(csv_folder_path, file)
    if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
        pattern = r'([^_]+)_(\w+)_\d+\.csv'
        match = re.match(pattern, file)
        print(file)
        if match:
            parameter_value = match.group(1).replace('_', '.')
            parameter_type = match.group(2)
        df_current = pd.read_csv(file_path)
        for x in df_current.index:
            df_current['parameter_type', x] = parameter_type
            df_current['parameter_value', x] = parameter_value
        print(df_current.to_string())
        pd.concat([df, df_current], ignore_index=True)

final_csv_file_path = os.path.join(csv_folder_path, '001final.csv')