import leaf_area as la
from config import paths
import os

folder_paths = paths.get_paths()

# Set folders
corrected_folder_path = folder_paths["corrected"]
print(corrected_folder_path)

marching_cubes_value = 0.3
ply_file_path = 'G:/My Drive/Dmitrii - Ph.D Thesis/Frost room Experiment Data/LA/plys/corrected_ready_to_execute/030723_11_2_2500p23.ply'
marching_cubes_shape = la.create_marching_cubes_shape(ply_file_path, marching_cubes_value, output_file_path='mcube.ply')