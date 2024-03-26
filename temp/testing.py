from config import paths
import pandas as pd
import os
import re

# Set path
folder_paths = paths.get_paths()


corrected_folder_path = folder_paths["corrected"]

plys = os.listdir(corrected_folder_path)
df = pd.DataFrame()
parameters = {}
for file in plys:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        match = re.search(r'(\d+p\d+\.)', file)
        parameters['Measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
        parameters['File_name'] = file
        if parameters['Measured_leaf_area'] > 0:
            df = pd.concat([df, pd.DataFrame([parameters])], ignore_index=True)

print(df.to_string())