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
poisson_folder_path = folder_paths["poissons"]
csv_folder_path = folder_paths["plys"]
csv_file_name = "poissons.csv"
csv_file_path = os.path.join(csv_folder_path, csv_file_name)

plys = os.listdir(corrected_folder_path)
df = pd.DataFrame()

# Start time measurement
start_time = time.time()

for file in plys:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set up iteration start time
        iteration_time = time.time()

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

        # Measure time taken for this iteration
        iteration_time = time.time() - iteration_time
        print(f"Time taken for this iteration: {iteration_time} seconds")

print(df.to_string())
df.to_csv(csv_file_path, index=False)

# Total time taken for the loop
total_time = time.time() - start_time
print(f"Total time taken: {total_time} seconds")