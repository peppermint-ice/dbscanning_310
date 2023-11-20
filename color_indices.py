""" This file contains functions regarding working with colors"""
from sklearn.metrics import pairwise_distances
import numpy as np
import db_clusterization
import get_camera_info


def green_index(point_cloud):
    vertices = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    # Find indices of the dots with Green Index > 0.8
    correct_green_index = np.all((colors[1] - colors[0]) / (colors[1] + colors[0]) > 0.5, axis=1)
    filtered_points = vertices[correct_green_index]
    filtered_colors = colors[correct_green_index]
    return filtered_points, filtered_colors


ply_file_path = 'clustered.ply'
exported_file_path = 'color_filtered.ply'


point_cloud_array, point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
clean_points, clean_colors = green_index(point_cloud_file)
get_camera_info.basic_export(clean_points, clean_colors, exported_file_path)
