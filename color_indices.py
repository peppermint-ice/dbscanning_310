""" This file contains functions regarding working with colors"""
from sklearn.metrics import pairwise_distances
import numpy as np
import db_clusterization
import get_camera_info
import os


def green_index(point_cloud):
    vertices = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    print(colors.shape)

    correct_green_indices = []
    # Find indices of the dots with Green Index > 0.8
    for i in range(len(colors)):
        green_index_value = (colors[i, 1] - colors[i, 0]) / (colors[i, 1] + colors[i, 0])
        if green_index_value > -0.02:
            correct_green_indices.append(i)
    filtered_points = vertices[correct_green_indices]
    filtered_colors = colors[correct_green_indices]
    return filtered_points, filtered_colors


def red_index(point_cloud):
    vertices = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    print(colors.shape)

    correct_red_indices = []
    # Find indices of the dots with Green Index > 0.8
    for i in range(len(colors)):
        red_index_value = (colors[i, 0] - colors[i, 1]) / (colors[i, 1] + colors[i, 0])
        if red_index_value > 0.25:
            correct_red_indices.append(i)
    filtered_points = vertices[correct_red_indices]
    filtered_colors = colors[correct_red_indices]
    return filtered_points, filtered_colors


# To run for one file

# ply_file_path = 'clustered.ply'
# exported_file_path_green = 'color_filtered_green.ply'
# exported_file_path_red = 'color_filtered_red.ply'
#
# point_cloud_array, point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
# green_points, green_colors = green_index(point_cloud_file)
# get_camera_info.basic_export(green_points, green_colors, exported_file_path_green)
#
# red_points, red_colors = red_index(point_cloud_file)
# get_camera_info.basic_export(red_points, red_colors, exported_file_path_red)

# To run for the whole folder

ply_folder_path = r'D:\results\plys\selected\clipped'
export_red_folder_path = r'D:\results\plys\selected\clipped\color_filtered\red'
export_green_folder_path = r'D:\results\plys\selected\clipped\color_filtered\green'
plys = os.listdir(ply_folder_path)

for file in plys:
    ply_file_path = os.path.join(ply_folder_path, file)
    if os.path.isfile(ply_file_path):
        exported_file_path_red = os.path.join(export_red_folder_path, file)
        exported_file_path_green = os.path.join(export_green_folder_path, file)
        print(ply_file_path)
        print(exported_file_path_red)
        print(exported_file_path_green)
        point_cloud_array, point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
        green_points, green_colors = green_index(point_cloud_file)
        red_points, red_colors = red_index(point_cloud_file)
        get_camera_info.basic_export(red_points, red_colors, exported_file_path_red)
        get_camera_info.basic_export(green_points, green_colors, exported_file_path_green)

