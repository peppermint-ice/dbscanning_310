""" This file contains functions regarding working with camera points"""
from sklearn.metrics import pairwise_distances
import numpy as np
import open3d as o3d
import db_clusterization


def extract_camera_points(point_cloud):
    """ This function returns an array with points of only one color that represent the camera color"""
    # Creating arrays containing data about points and colors
    vertices = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    # Define the color
    camera_color = [0, 1, 0]
    # Find indices of the dots with the defined color & then extract these points
    camera_indices = np.all(colors == camera_color, axis=1)
    camera_points = vertices[camera_indices]
    return camera_points


def export_camera_points(camera_points):
    """ This is basically another export functon. Just easier than the frst one + no colors"""
    clustered_point_cloud = o3d.geometry.PointCloud()
    clustered_point_cloud.points = o3d.utility.Vector3dVector(camera_points)
    o3d.io.write_point_cloud('camera_points.ply', clustered_point_cloud, write_ascii=True)


def create_a_sphere(camera_points):
    central_point = np.mean(camera_points, axis=0)
    distances = pairwise_distances(camera_points, [central_point])
    # print(distances)
    distance_95 = np.percentile(distances, 95)
    distance = distance_95 * 1.1
    return [central_point, distance]


def crop_a_point_cloud(point_cloud, sphere):
    vertices = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    # Extract the sphere parameters
    sphere_center, sphere_radius = sphere

    # Calculate the squared radius for efficient comparison
    sphere_radius_squared = sphere_radius ** 2

    # Calculate the squared distance from each point to the sphere center
    distances_squared = np.sum((vertices - sphere_center) ** 2, axis=1)

    # Find the indices of points inside the sphere
    indices_inside_sphere = np.where(distances_squared <= sphere_radius_squared)[0]

    # Extract the points and colors inside the sphere
    new_vertices = vertices[indices_inside_sphere]
    new_colors = colors[indices_inside_sphere]

    return new_vertices, new_colors


def basic_export(new_vertices, new_colors, output_filename):
    # Create a new PointCloud object
    cropped_point_cloud = o3d.geometry.PointCloud()

    # Set the points and colors of the cropped PointCloud
    cropped_point_cloud.points = o3d.utility.Vector3dVector(new_vertices)
    cropped_point_cloud.colors = o3d.utility.Vector3dVector(new_colors)

    # Save the cropped PointCloud to a PLY file in ASCII format
    o3d.io.write_point_cloud(output_filename, cropped_point_cloud, write_ascii=True)


ply_file_path = 'sfm.ply'
exported_file_path = 'camera_sphere.ply'
exported_point_cloud = 'cropped_point_cloud.ply'

point_cloud_array, point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
cam_points = extract_camera_points(point_cloud_file)
# export_camera_points(cam_points)
sphere_parameters = create_a_sphere(cam_points)
# print(sphere_parameters)
cropped_vertices, cropped_colors = crop_a_point_cloud(point_cloud_file, sphere_parameters)
basic_export(cropped_vertices, cropped_colors, exported_point_cloud)
