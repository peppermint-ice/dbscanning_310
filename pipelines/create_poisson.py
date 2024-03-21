from dbscanning_310 import leaf_area as la
from dbscanning_310.config import paths
import os
import re
import pandas as pd


# Set import path
folder_paths = paths.get_paths()

# Set export folders
corrected_folder_path = folder_paths["corrected"]
poisson_folder_path = folder_paths["poissons"]
csv_folder_path = folder_paths["plys"]
csv_file_name = "poissons.csv"
csv_file_path = os.path.join(csv_folder_path, csv_file_name)

plys = os.listdir(corrected_folder_path)
df = pd.DataFrame()
for file in plys:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set export folder for poisson shapes
        poisson_export_file_path = os.path.join(poisson_folder_path, file)
        print(ply_file_path)
        print(poisson_export_file_path)

        # Open a ply
        pcl = la.open_ply_file(ply_file_path)

        # Create poisson shapes
        poisson_value = 12  # Adjust poisson depth as needed
        poisson_shape = la.create_poisson_shape(ply_file_path, poisson_value, poisson_export_file_path)
        total_volume = la.calculate_watertight_volume(poisson_shape)

        # Remember parameters
        parameters = la.calculate_shape_parameters(pcl, poisson_shape, total_volume)
        parameters['File_name'] = file
        match = re.search(r'(\d+p\d+\.)', file)
        parameters['Measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
        if parameters['Measured_leaf_area'] > 0:
            df = pd.concat([df, pd.DataFrame([parameters])], ignore_index=True)

print(df.to_string())
df.to_csv(csv_file_path, index=False)