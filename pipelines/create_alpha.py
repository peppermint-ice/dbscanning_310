import leaf_area as la
from config import paths
import os
import re
import pandas as pd
import time
import sys

# Extract start and end indices from command-line arguments
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])


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

# Start time measurement
start_time = time.time()

for file in plys[start_index-1:end_index]:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set up iteration start time
        iteration_time = time.time()

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

        # Measure time taken for this iteration
        iteration_time = time.time() - iteration_time
        print(f"Time taken for this iteration: {iteration_time} seconds")

print(df.to_string())
df.to_csv(csv_file_path, index=False)

# Total time taken for the loop
total_time = time.time() - start_time
print(f"Total time taken: {total_time} seconds")