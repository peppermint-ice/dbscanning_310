from config import paths
import os
import re
import pandas as pd


# Get path
folder_paths = paths.get_paths()
csv_folder_path = folder_paths["hyperparameters"]
df = pd.DataFrame()
for file in os.listdir(csv_folder_path):
    if file.endswith(".csv"):
        filepath = os.path.join(csv_folder_path, file)
        current_df = pd.read_csv(filepath)
        df = pd.concat([df, current_df], ignore_index=True)
        print(f"df {file} added")
print(df.to_string())