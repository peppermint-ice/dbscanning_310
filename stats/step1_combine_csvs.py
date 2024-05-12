from config import paths
import os
import re
import pandas as pd

# Get path
folder_paths = paths.get_paths()

# Set folder
csv_folder_path = folder_paths["hyperparameters"]

csvs = os.listdir(csv_folder_path)

df = pd.DataFrame()

for file in csvs:
    file_path = os.path.join(csv_folder_path, file)
    if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
        pattern = r'(\d+_?\d*)(\D+)(\d+)\.csv'
        match = re.match(pattern, file)
        print(file)
        if match:
            parameter_value = match.group(1).replace('_', '.')
            parameter_type = match.group(2)
            print(parameter_type)
            print(parameter_value)
        df_current = pd.read_csv(file_path)

        for x in df_current.index:
            df_current.loc[x, 'parameter_type'] = parameter_type
            df_current.loc[x, 'parameter_value'] = parameter_value
        print(df_current.to_string())
        df = pd.concat([df, df_current], ignore_index=True)

final_csv_file_path = os.path.join(csv_folder_path, '002final.csv')
df.to_csv(final_csv_file_path, index=False)