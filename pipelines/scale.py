from dbscanning_310 import leaf_area as la
from dbscanning_310.config import paths
import os

folder_paths = paths.get_paths()

circles_folder_path = folder_paths["circles"]
green_folder_path = folder_paths["green_cubes"]
rotated_export_folder_path = folder_paths["rotated_cubes"]
plys = os.listdir(circles_folder_path)

for file in plys:
    ply_file_path = os.path.join(circles_folder_path, file)
    green_ply_file_path = os.path.join(green_folder_path, file)
    rotated_export_file_path = os.path.join(rotated_export_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        print(ply_file_path)
        print(rotated_export_file_path)
        print(green_ply_file_path)

        pcl = la.open_ply_file(ply_file_path)
        circle_pcl = pcl
        green_pcl = la.open_ply_file(green_ply_file_path)

        scaling_parameters = la.calculate_rotation_and_scaling(circle_pcl)
        vertices, colors = la.transform_point_cloud(green_pcl, scaling_parameters)
        la.export_ply_file(vertices, colors, output_filepath=rotated_export_file_path)