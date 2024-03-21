from dbscanning_310 import leaf_area as la
from dbscanning_310.config import paths
import os
import re
import pandas as pd


# Set import path
folder_paths = paths.get_paths()

# Set export folders
corrected_folder_path = folder_paths["corrected"]
alpha_folder_path = folder_paths["alphas"]
csv_folder_path = folder_paths["data"]
csv_file_name = "alphas.csv"
csv_file_path = os.path.join(csv_folder_path, csv_file_name)


plys = os.listdir(corrected_folder_path)
df = pd.DataFrame()
for file in plys:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set export folder for alpha shapes
        alpha_export_file_path = os.path.join(alpha_folder_path, file)
        print(ply_file_path)
        print(alpha_export_file_path)

        # Open a ply
        pcl = la.open_ply_file(ply_file_path)

        # Create alpha shapes
        alpha_value = 0.5  # Adjust alpha as needed
        alpha_shape = la.create_alpha_shape(ply_file_path, alpha_value, alpha_export_file_path)
        total_volume = la.calculate_watertight_volume(alpha_shape)

        # Remember parameters
        parameters = la.calculate_shape_parameters(pcl, alpha_shape, total_volume)
        parameters['File_name'] = file
        match = re.search(r'(\d+p\d+\.)', file)
        parameters['Measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
        if parameters['Measured_leaf_area'] > 0:
            df = pd.concat([df, pd.DataFrame([parameters])], ignore_index=True)

print(df.to_string())
df.to_csv(csv_file_path, index=False)