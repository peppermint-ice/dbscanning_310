from config import paths
import os
import re
import pandas as pd

# Get path
folder_paths = paths.get_paths()

# Set folder
csv_folder_path = folder_paths["multilinear_results"]
csv_export_path = os.path.join(folder_paths["data"], "combine_multilinear.csv")
csvs = os.listdir(csv_folder_path)

df = pd.DataFrame()

for file in csvs:
    file_path = os.path.join(csv_folder_path, file)
    if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
        print(file)
        df_current = pd.read_csv(file_path)
        if 'noyear' in file:
            df_current.loc[:, 'by_year'] = False
        else:
            df_current.loc[:, 'by_year'] = True
        df = pd.concat([df, df_current], ignore_index=True)

    print('file_scanned')

df.to_csv(csv_export_path, index=False)