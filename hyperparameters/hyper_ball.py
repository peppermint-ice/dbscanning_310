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


# Get path
folder_paths = paths.get_paths()

# Set folders
corrected_folder_path = folder_paths["corrected"]
ball_pivoting_folder_path = folder_paths["ball_pivoting"]
csv_folder_path = folder_paths["hyperparameters"]


plys = os.listdir(corrected_folder_path)


# Select desired ball_pivoting values

ball_pivoting_values = [[0.05, 0.1, 0.2, 0.4],
                        [0.1, 0.2, 0.4, 0.8],
                        [0.152, 0.3, 0.61, 1.2],
                        [0.2, 0.4, 0.8, 1.6],
                        [0.25, 0.5, 1.0, 2.0],
                        [0.3, 0.6, 1.2, 2.4],
                        [0.35, 0.7, 1.4, 2.8],
                        [0.4, 0.8, 1.6, 3.2],
                        [0.45, 0.9, 1.8, 3.6],
                        [0.5, 1.0, 2.0, 4.0]]

# Start time measurement
start_time = time.time()

for ball_pivoting_value in ball_pivoting_values:
    df = pd.DataFrame()
    for file in plys[start_index - 1:end_index]:
        ply_file_path = os.path.join(corrected_folder_path, file)
        print("ball_pivoting value: " + str(ball_pivoting_value))
        if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
            # Set up iteration start time
            iteration_time = time.time()

            try:
                # Open a ply
                pcl = la.open_ply_file(ply_file_path)

                # Define export path
                value_folder_path = os.path.join(ball_pivoting_folder_path, str(ball_pivoting_value[0]))
                os.makedirs(value_folder_path, exist_ok=True)
                mesh_export_path = os.path.join(value_folder_path, ply_file_path)

                # Create ball_pivoting shapes
                ball_pivoting_shape = la.create_ball_pivoting_shape(ply_file_path, ball_pivoting_value, mesh_export_path)
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

                csv_file_name = str(ball_pivoting_value[0]).replace(".", "_") + "ball_pivotings" + str((start_index - 1)) + '.csv'
                csv_file_path = os.path.join(csv_folder_path, csv_file_name)

                print(df.to_string())
            except IndexError:
                print("Index error. The ball_pivoting is not good")
                continue
        df.to_csv(csv_file_path, index=False)

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")