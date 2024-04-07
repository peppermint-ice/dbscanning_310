import pandas as pd
import os
from config import paths


# Get path
folder_paths = paths.get_paths()
csv_import_path = os.path.join(folder_paths["data"], '001final.csv')
csv_folder_path = folder_paths["hyperparameters"]


# Load the dataset into a pandas DataFrame
data = pd.read_csv(csv_import_path)


# Extracting year from the 'File_name' column
data['Year'] = '20' + data['File_name'].str[:6].str[-2:]
# Drop the 'File_name' column
data.drop(columns=['File_name'], inplace=True)


for parameter_type in data['parameter_type'].unique():
    df_param = data[data['parameter_type'] == parameter_type]
    for parameter_value in df_param['parameter_value'].unique():
        df_current = df_param[df_param['parameter_value'] == parameter_value]
        file_name = str(parameter_type) + str(parameter_value).replace('.', '_') + '.csv'
        csv_export_path = os.path.join(csv_folder_path, file_name)
        print(file_name, len(df_current))
        df_current.drop(columns=['Compactness'], inplace=True)
        df_current.drop(columns=['parameter_type'], inplace=True)
        if len(df_current) > 210:
            df_current.to_csv(csv_export_path, index=False)