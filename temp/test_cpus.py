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


        # Measure time taken for this iteration
        iteration_time = time.time() - iteration_time
        print("Time taken for this iteration: " + str(iteration_time) + " seconds")

        csv_file_name = "alphas" + str((start_index - 1)) + '.csv'
        csv_file_path = os.path.join(csv_folder_path, csv_file_name)


# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")