import numpy as np
import db_clusterization
import get_camera_info
from skimage.measure import ransac
from skimage.draw import disk
from scipy.spatial.transform import Rotation
import open3d as o3d

import numpy as np
from skimage.measure import ransac
from skimage.draw import disk
import open3d as o3d
import numpy as np
from skimage.measure import ransac, CircleModel
from skimage.draw import disk

import numpy as np
from skimage.measure import ransac, CircleModel
import open3d as o3d


def fit_circle_to_point_cloud_and_save_ply(point_cloud, output_ply_path, inlier_threshold=0.1, max_iterations=1000):
    """
    Fits a circle to a 3D point cloud using RANSAC and saves the point cloud and the fitted circle as .ply files.

    Parameters:
    - point_cloud: numpy array of shape (N, 3) representing 3D points in the point cloud.
    - output_ply_path: path to save the .ply file.
    - inlier_threshold: threshold for considering a point as an inlier during RANSAC.
    - max_iterations: maximum number of RANSAC iterations.

    Returns:
    - center: estimated center of the fitted circle.
    - radius: estimated radius of the fitted circle.
    - normal: normal vector of the plane containing the circle.
    """
    # Use RANSAC to fit the circle
    _, inliers = ransac(point_cloud[:, :2], CircleModel, min_samples=3,
                        residual_threshold=inlier_threshold, max_trials=max_iterations)

    # Extract circle parameters (center and radius)
    model = CircleModel()
    center = model.estimate(point_cloud[inliers])
    radius = model.radius

    # Calculate the normal vector of the plane containing the circle
    normal_vector = np.cross(point_cloud[inliers, :] - center, point_cloud[inliers[-1:] + [0], :] - center)
    normal_vector /= np.linalg.norm(normal_vector)

    # Save the point cloud and the fitted circle as .ply files
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

    return center, radius, normal_vector


def find_top_of_pot(point_cloud, inlier_threshold=0.01, max_iterations=1000):
    """
    Finds the coordinates of the top of the pot by fitting a horizontal plane using RANSAC.

    Parameters:
    - point_cloud: numpy array of shape (N, 3) representing 3D points in the point cloud.
    - inlier_threshold: threshold for considering a point as an inlier during RANSAC.
    - max_iterations: maximum number of RANSAC iterations.

    Returns:
    - top_coordinates: coordinates of the estimated top of the pot.
    """
    # Convert the numpy array to an Open3D PointCloud
    pot_pcd = o3d.geometry.PointCloud()
    pot_pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Use RANSAC to fit a plane to the top of the pot
    plane_model, inliers = pot_pcd.segment_plane(distance_threshold=inlier_threshold,
                                                 ransac_n=max_iterations)

    # Extract plane parameters (normal vector and distance from the origin)
    normal_vector = np.array(plane_model[:3])
    distance = plane_model[3]

    # Calculate the centroid of the inlier points as the top coordinates
    inlier_points = np.asarray(pot_pcd.points)[inliers]
    top_coordinates = np.mean(inlier_points, axis=0)

    return top_coordinates

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

# pot_center, pot_radius, pot_normal = fit_circle_to_point_cloud_and_save_ply(pot_point_cloud_array, 'output.ply')
pot_top = find_top_of_pot(pot_point_cloud_array)