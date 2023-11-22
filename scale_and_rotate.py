import numpy as np
import db_clusterization
import get_camera_info
from skimage.measure import ransac
from skimage.draw import disk
from scipy.spatial.transform import Rotation
import open3d as o3d


def fit_circle_to_point_cloud_and_save_ply(point_cloud, output_ply_path, inlier_threshold=0.1, max_iterations=1000):
    """
    Fits a circle to a 3D point cloud using RANSAC and saves the point cloud and the fitted circle as a .ply file.

    Parameters:
    - point_cloud: numpy array of shape (N, 3) representing 3D points in the point cloud.
    - output_ply_path: path to save the .ply file.
    - inlier_threshold: threshold for considering a point as an inlier during RANSAC.
    - max_iterations: maximum number of RANSAC iterations.

    Returns:
    - center: estimated center of the fitted circle.
    - radius: estimated radius of the fitted circle.
    - normal: estimated normal vector of the plane containing the circle.
    """
    # Define the circle model for RANSAC
    model, inliers = ransac(point_cloud, ransac.circle, min_samples=3,
                            residual_threshold=inlier_threshold, max_trials=max_iterations)

    # Extract circle parameters (center and radius)
    center, radius = model.params

    # Calculate the normal vector of the plane containing the circle
    normal = np.cross(point_cloud[inliers[0], :] - center, point_cloud[inliers[1], :] - center)
    normal /= np.linalg.norm(normal)

    # Save the point cloud and the fitted circle as a .ply file
    pot_pcd = o3d.geometry.PointCloud()
    pot_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(output_ply_path, pot_pcd)

    # Create the fitted circle using Open3D
    fitted_circle_points = []
    for theta in np.linspace(0, 2 * np.pi, 100):
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2]
        fitted_circle_points.append([x, y, z])

    fitted_circle_pcd = o3d.geometry.PointCloud()
    fitted_circle_pcd.points = o3d.utility.Vector3dVector(fitted_circle_points)
    o3d.io.write_point_cloud(output_ply_path.replace('.ply', '_fitted_circle.ply'), fitted_circle_pcd)

    return center, radius, normal



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

