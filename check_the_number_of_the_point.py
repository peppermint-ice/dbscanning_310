from db_clusterization import open_ply_file
import numpy as np
import open3d as o3d


def fit_plane_least_squares(points):
    """
    Fits a plane to a set of 3D points using least squares analysis.

    Parameters:
    - points: numpy array of shape (N, 3) representing 3D points.

    Returns:
    - plane_coeffs: coefficients of the plane equation [a, b, c, d] for the equation ax + by + cz + d = 0.
    """
    # Augment the points with a column of ones for the bias term
    augmented_points = np.column_stack((points, np.ones(len(points))))

    # Solve the least squares problem to find the coefficients [a, b, c, d]
    plane_coeffs, _, _, _ = np.linalg.lstsq(augmented_points[:, :3], -augmented_points[:, 3], rcond=None)

    return plane_coeffs


def rotate_ply_to_xy_and_save(plane_coeffs, point_cloud, save_path="rotated_pot.ply"):
    """
    Rotates a point cloud to align the given plane with the XY plane and saves it as a PLY file.

    Parameters:
    - plane_coeffs: coefficients of the plane equation [a, b, c, d] for the equation ax + by + cz + d = 0.
    - point_cloud: numpy array of shape (N, 3) representing 3D points.
    - save_path: path to save the rotated point cloud as a PLY file.

    Returns:
    - rotated_point_cloud: rotated point cloud.
    """
    # Calculate the normal vector of the plane
    normal_vector = plane_coeffs[:3]

    # Calculate the rotation matrix to align the normal vector with the Z-axis
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, normal_vector)
    rotation_angle = np.arccos(np.dot(z_axis, normal_vector) / (np.linalg.norm(z_axis) * np.linalg.norm(normal_vector)))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)

    # Apply the rotation to the point cloud
    rotated_point_cloud = np.dot(rotation_matrix, point_cloud.T).T

    # Create an Open3D PointCloud object from the rotated point cloud
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_point_cloud)

    # Save the rotated point cloud as a PLY file
    o3d.io.write_point_cloud(save_path, rotated_pcd)

    return rotated_point_cloud

# Example usage:
# Assuming you have a point cloud stored in a variable called 'pot_point_cloud'
# Manually enter the indices of the points you want to use for fitting the plane
selected_indices = [6, 941, 206, 2214, 842, 671, 323, 97, 333, 758, 66]

pot_point_cloud, pot_file = open_ply_file("pot_clustered.ply")

# Extract the subset of points based on the selected indices
subset_of_points = pot_file.points[selected_indices]

# Fit the plane to the subset of points using least squares
plane_coeffs = fit_plane_least_squares(subset_of_points)

# Check the orientation of the plane based on the z-coordinates
if np.sum(subset_of_points[:, 2] > 0) > np.sum(subset_of_points[:, 2] < 0):
    # If there are more points with positive z-coordinates, flip the normal vector
    plane_coeffs[:3] *= -1

# Rotate the point cloud to align the plane with the XY plane and save it
rotated_point_cloud = rotate_ply_to_xy_and_save(plane_coeffs, pot_point_cloud, 'rotated_pot.ply')

