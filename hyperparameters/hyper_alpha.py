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
alpha_folder_path = folder_paths["alphas"]
csv_folder_path = folder_paths["hyperparameters"]


plys = os.listdir(corrected_folder_path)
df = pd.DataFrame()

# Select desired alpha values

alpha_values = [0.001, 0.005, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 7.5, 10, 15, 20, 25, 50, 100, 1000]

# Start time measurement
start_time = time.time()

for file in plys[start_index-1:end_index]:
    ply_file_path = os.path.join(corrected_folder_path, file)
    for alpha_value in alpha_values:
        print("Alpha value: " + str(alpha_value))
        if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
            # Set up iteration start time
            iteration_time = time.time()

            # Open a ply
            pcl = la.open_ply_file(ply_file_path)

            # Create alpha shapes
            alpha_shape = la.create_alpha_shape(ply_file_path, alpha_value)
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
            print("Time taken for this iteration: " + str(iteration_time) + " seconds")

            csv_file_name = str(alpha_value).replace(".", "_") + "alphas" + str((start_index - 1)) + '.csv'
            csv_file_path = os.path.join(csv_folder_path, csv_file_name)

            print(df.to_string())
            df.to_csv(csv_file_path, index=False)

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")