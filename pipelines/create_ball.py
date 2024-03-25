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
ball_pivoting_folder_path = folder_paths["ball_pivoting"]
csv_folder_path = folder_paths["data"]

plys = os.listdir(corrected_folder_path)
df = pd.DataFrame()

# Start time measurement
start_time = time.time()

for file in plys:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set up iteration start time
        iteration_time = time.time()

        # Set export folder for ball_pivoting shapes
        ball_pivoting_export_file_path = os.path.join(ball_pivoting_folder_path, file)
        print(ply_file_path)
        print(ball_pivoting_export_file_path)

        # Open a ply
        pcl = la.open_ply_file(ply_file_path)

        # Create ball_pivoting shapes
        radii_value = [3, 10, 15, 20]  # Adjust ball_pivoting depth as needed
        ball_pivoting_shape = la.create_ball_pivoting_shape(ply_file_path, radii_value, ball_pivoting_export_file_path)
        total_volume = la.calculate_watertight_volume(ball_pivoting_shape)

        # Remember parameters
        parameters = la.calculate_shape_parameters(pcl, ball_pivoting_shape, total_volume)
        parameters['File_name'] = file
        match = re.search(r'(\d+p\d+\.)', file)
        parameters['Measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
        if parameters['Measured_leaf_area'] > 0:
            df = pd.concat([df, pd.DataFrame([parameters])], ignore_index=True)

        # Measure time taken for this iteration
        iteration_time = time.time() - iteration_time
        print("Time taken for this iteration: " + str(iteration_time) + " seconds")

        csv_file_name = "ball" + str((start_index - 1)) + '.csv'
        csv_file_path = os.path.join(csv_folder_path, csv_file_name)

        print(df.to_string())
        df.to_csv(csv_file_path, index=False)

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")