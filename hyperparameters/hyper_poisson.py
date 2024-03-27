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
poisson_folder_path = folder_paths["poissons"]
csv_folder_path = folder_paths["hyperparameters"]


plys = os.listdir(corrected_folder_path)


# Select desired poisson values

poisson_values = [8, 9, 10, 11, 12, 13, 14, 15]

# Start time measurement
start_time = time.time()

for poisson_value in poisson_values:
    df = pd.DataFrame()
    for file in plys[start_index - 1:end_index]:
        ply_file_path = os.path.join(corrected_folder_path, file)
        print("poisson value: " + str(poisson_value))
        if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
            # Set up iteration start time
            iteration_time = time.time()

            try:
                # Open a ply
                pcl = la.open_ply_file(ply_file_path)

                # Create poisson shapes
                poisson_shape = la.create_poisson_shape(ply_file_path, poisson_value)
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
                print("Time taken for this iteration: " + str(iteration_time) + " seconds")

                csv_file_name = str(poisson_value).replace(".", "_") + "poissons" + str((start_index - 1)) + '.csv'
                csv_file_path = os.path.join(csv_folder_path, csv_file_name)

                print(df.to_string())
            except IndexError:
                print("Index error. The poisson is not good")
                continue
        df.to_csv(csv_file_path, index=False)

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")