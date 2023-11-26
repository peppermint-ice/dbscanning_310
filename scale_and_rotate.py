import numpy as np
import db_clusterization
import get_camera_info
from skimage.measure import ransac
from skimage.draw import disk
from scipy.spatial.transform import Rotation
import open3d as o3d
import os
from scipy.optimize import minimize


def fit_plane_least_squares(points):
    """
    Fit a plane to 3D points using least squares.

    Parameters:
        points (numpy array): 2D array of points with shape (n, 3).

    Returns:
        normal_vector (numpy array): Normal vector of the fitted plane.
    """

    # Define the objective function for least squares plane fitting
    def objective_function(params):
        a, b, c, d = params
        return np.sum((a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) ** 2)

    # Initial guess for the plane parameters
    initial_guess = np.ones(4)

    # Minimize the objective function to find the plane parameters
    result = minimize(objective_function, initial_guess, method='trust-constr')

    # Extract the normal vector of the fitted plane
    normal_vector = result.x[:3] / np.linalg.norm(result.x[:3])

    return normal_vector


def calculate_rotation_and_scaling(circle_point_cloud):
    # Convert input to NumPy arrays
    points = np.array(circle_point_cloud.points)

    # Fit a plane to the points in the circle point cloud
    normal_vector = fit_plane_least_squares(points)

    # Get the rotation matrix to align the plane with the XY plane
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(normal_vector, z_axis)
    rotation_angle = np.arccos(np.dot(normal_vector, z_axis))
    rotation_matrix = Rotation.from_rotvec(rotation_vector * rotation_angle).as_matrix()

    # Rotate the entire circle point cloud
    rotated_points = np.dot(points, rotation_matrix.T)

    # Project the rotated points onto the XY plane
    projected_points = rotated_points[:, :2]

    # Fit a circle to the 2D points on the XY plane
    fitted_circle_center, fitted_circle_radius = fit_circle_least_squares(projected_points)

    # Calculate the scaling factor based on the diameter of the fitted circle
    scale_factor = 15 / (2 * fitted_circle_radius)

    return rotation_matrix, scale_factor


def fit_circle_least_squares(points):
    """
    Fit a circle to 2D points using least squares.

    Parameters:
        points (numpy array): 2D array of points with shape (n, 2).

    Returns:
        circle_center (numpy array): Center coordinates of the fitted circle.
        circle_radius (float): Radius of the fitted circle.
    """
    x, y = points[:, 0], points[:, 1]

    # Initial guess for the circle parameters
    initial_guess = np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2))

    # Define the objective function for least squares fitting
    objective_function = lambda params: np.sum((np.sqrt((x - params[0]) ** 2 + (y - params[1]) ** 2) - params[2]) ** 2)

    # Minimize the objective function to find the circle parameters
    result = minimize(objective_function, initial_guess, method='trust-constr')

    # Extract the fitted circle parameters
    circle_center = result.x[:2]
    circle_radius = result.x[2]

    return circle_center, circle_radius


def transform_point_cloud(point_cloud, rotation_matrix, scale_factor):
    # Convert input to NumPy arrays
    points = np.array(point_cloud.points)

    # Translate the point cloud so that the circle center is at the origin
    translated_point_cloud = points - np.mean(points, axis=0)

    # Apply rotation to the translated point cloud
    rotated_point_cloud = np.dot(translated_point_cloud, rotation_matrix.T)

    # Apply scaling to the rotated point cloud
    scaled_point_cloud = rotated_point_cloud * scale_factor

    # Extract original colors
    colors = np.array(point_cloud.colors)

    return scaled_point_cloud, colors


# circle_file_path = "circle.ply"
# ply_file_path = "clustered.ply"
# pot_file_path = "pot_clustered.ply"
# plant_file_path = "color_filtered_green.ply"

# exported_ply_file_path = "rotated_clustered.ply"
# exported_pot_file_path = "rotated_red.ply"
# exported_plant_file_path = "rotated_green.ply"

# circle_point_cloud_array, circle_point_cloud_file = db_clusterization.open_ply_file(circle_file_path)
# ply_point_cloud_array, ply_point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
# pot_point_cloud_array, pot_point_cloud_file = db_clusterization.open_ply_file(pot_file_path)
# plant_point_cloud_array, plant_point_cloud_file = db_clusterization.open_ply_file(plant_file_path)
#
#
# rotation_matrix, scale_factor = calculate_rotation_and_scaling(circle_point_cloud_file)
# transformed_point_cloud, transformed_colors = transform_point_cloud(ply_point_cloud_file, rotation_matrix, scale_factor)
# get_camera_info.basic_export(transformed_point_cloud, transformed_colors, exported_ply_file_path)


# To run for the whole folder

circles_folder_path = r'D:\results\plys\selected\circles'

clipped_folder_path = r'D:\results\plys\selected\clipped'
plant_folder_path = r'D:\results\plys\selected\clipped\color_filtered\green'
clipped_clustered_folder_path = r'D:\results\plys\selected\clipped\clustered'
clustered_plant_folder_path = r'D:\results\plys\selected\clipped\clustered\color_filtered\green'

export_clipped_folder_path = r'D:\results\plys\selected\clipped\rotated'
export_plant_folder_path = r'D:\results\plys\selected\clipped\color_filtered\green\rotated'
export_clipped_clustered_folder_path = r'D:\results\plys\selected\clipped\clustered\rotated'
export_clustered_plant_folder_path = r'D:\results\plys\selected\clipped\clustered\color_filtered\green\rotated'


plys = os.listdir(circles_folder_path)
for file in plys:
    circle_file_path = os.path.join(circles_folder_path, file)
    if os.path.isfile(circle_file_path):

        clipped_file_path = os.path.join(clipped_folder_path, file)
        plant_file_path = os.path.join(plant_folder_path, file)
        clipped_clustered_file_path = os.path.join(clipped_clustered_folder_path, file)
        clustered_plant_file_path = os.path.join(clustered_plant_folder_path, file)

        export_clipped_file_path = os.path.join(export_clipped_folder_path, file)
        export_plant_file_path = os.path.join(export_plant_folder_path, file)
        export_clipped_clustered_file_path = os.path.join(export_clipped_clustered_folder_path, file)
        export_clustered_plant_file_path = os.path.join(export_clustered_plant_folder_path, file)

        print("Circle: ", circle_file_path)

        print("Clipped: from")
        print(clipped_file_path)

        print("to:")
        print(export_clipped_file_path)
        print("Plant: from")
        print(plant_file_path)

        print("to:")
        print(export_plant_file_path)
        print("Clipped_clustered: from")
        print(clipped_clustered_file_path)

        print("to:")
        print(export_clipped_clustered_file_path)
        print("Clustered_plant: from")
        print(clustered_plant_file_path)

        print("to:")
        print(export_clustered_plant_file_path)

        circle_point_cloud_array, circle_point_cloud_file = db_clusterization.open_ply_file(circle_file_path)
        clipped_point_cloud_array, clipped_point_cloud_file = db_clusterization.open_ply_file(clipped_file_path)
        plant_point_cloud_array, plant_point_cloud_file = db_clusterization.open_ply_file(plant_file_path)
        clipped_clustered_point_cloud_array, clipped_clustered_point_cloud_file = db_clusterization.open_ply_file(
            clipped_clustered_file_path)
        clustered_plant_point_cloud_array, clustered_plant_point_cloud_file = db_clusterization.open_ply_file(
            clustered_plant_file_path)

        rotation_matrix, scale_factor = calculate_rotation_and_scaling(circle_point_cloud_file)

        transformed_clipped_point_cloud, transformed_clipped_colors = transform_point_cloud(
            clipped_point_cloud_file, rotation_matrix, scale_factor)
        transformed_plant_point_cloud, transformed_plant_colors = transform_point_cloud(
            plant_point_cloud_file, rotation_matrix, scale_factor)
        transformed_clipped_clustered_point_cloud, transformed_clipped_clustered_colors = transform_point_cloud(
            clipped_clustered_point_cloud_file, rotation_matrix, scale_factor)
        transformed_clustered_plant_point_cloud, transformed_clustered_plant_colors = transform_point_cloud(
            clustered_plant_point_cloud_file, rotation_matrix, scale_factor)

        get_camera_info.basic_export(transformed_clipped_point_cloud, transformed_clipped_colors,
                                     export_clipped_file_path)
        get_camera_info.basic_export(transformed_plant_point_cloud, transformed_plant_colors,
                                     export_plant_file_path)
        get_camera_info.basic_export(transformed_clipped_clustered_point_cloud, transformed_clipped_clustered_colors,
                                     export_clipped_clustered_file_path)
        get_camera_info.basic_export(transformed_clustered_plant_point_cloud, transformed_clustered_plant_colors,
                                     export_clustered_plant_file_path)
