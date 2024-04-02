import leaf_area as la
from config import paths
import os
import re
import pandas as pd
import time
import sys

# Get path
folder_paths = paths.get_paths()

# Set folders
corrected_folder_path = folder_paths["corrected"]
marching_cubes_folder_path = folder_paths["marching_cubes"]
csv_folder_path = folder_paths["data"]

plys = os.listdir(corrected_folder_path)


# Select desired marching_cubes values


# Start time measurement
start_time = time.time()

marching_cubes_value = 0.5

df = pd.DataFrame()
for file in plys:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set up iteration start time
        iteration_time = time.time()
        try:
            # Open a ply
            pcl = la.open_ply_file(ply_file_path)
            print("File loaded " + file)
            marching_cubes_export_file_path = os.path.join(marching_cubes_folder_path, file)
            # Create marching_cubes shapes
            marching_cubes_shape = la.create_marching_cubes_shape(ply_file_path, marching_cubes_value, marching_cubes_export_file_path)
            total_volume = la.calculate_watertight_volume(marching_cubes_shape)

            # Remember parameters
            parameters = la.calculate_shape_parameters(pcl, marching_cubes_shape, total_volume)
            parameters['File_name'] = file
            match = re.search(r'(\d+p\d+\.)', file)
            parameters['Measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
            if parameters['Measured_leaf_area'] > 0:
                df = pd.concat([df, pd.DataFrame([parameters])], ignore_index=True)

            # Measure time taken for this iteration
            iteration_time = time.time() - iteration_time
            print("Time taken for this iteration: " + str(iteration_time) + " seconds")

            csv_file_name = str(marching_cubes_value).replace(".", "_") + "marching_cubes" + '.csv'
            csv_file_path = os.path.join(csv_folder_path, csv_file_name)
            print("CSV path: " + csv_file_path)

        except IndexError:
            print("Index error. The marching_cubes is not good")
            continue
    df.to_csv(csv_file_path, index=False)
    print(df.to_string())
# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")