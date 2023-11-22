import numpy as np
import db_clusterization
import get_camera_info
from skimage.measure import ransac
from skimage.draw import disk
from scipy.spatial.transform import Rotation
import open3d as o3d






# export the original point cloud
# export the pot point cloud
# export the plant point cloud

ply_file_path = "clustered.ply"
pot_file_path = "color_filtered_red.ply"
plant_file_path = "color_filtered_green.ply"

exported_ply_file_path = "rotated_clustered.ply"
exported_pot_file_path = "rotated_red.ply"
exported_plant_file_path = "rotated_green.ply"

point_cloud_array, point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
pot_point_cloud_array, pot_point_cloud_file = db_clusterization.open_ply_file(pot_file_path)
plant_point_cloud_array, plant_point_cloud_file = db_clusterization.open_ply_file(plant_file_path)



get_camera_info.basic_export(rotated_points, rotated_colors, exported_ply_file_path)

get_camera_info.basic_export(rotated_points, rotated_colors, exported_pot_file_path)

get_camera_info.basic_export(rotated_points, rotated_colors, exported_plant_file_path)

