import numpy as np
import pandas as pd
import open3d as o3d

from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from config import paths

import os
import re
import random
import trimesh


def open_ply_file(file_path):
    """ This function opens a ply file and returns it as a numpy array """
    point_cloud_data_file = o3d.io.read_point_cloud(file_path)
    return point_cloud_data_file


def export_ply_file(vertices, colors=None, output_filepath=None):
    # Create a new PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Set the points and colors of the cropped PointCloud
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save the cropped PointCloud to a PLY file in ASCII format
    if output_filepath is not None:
        o3d.io.write_point_cloud(output_filepath, point_cloud, write_ascii=True)

    return point_cloud

def create_clusters(point_cloud_data_file, eps=0.15, min_samples=80):
    """
    This function requires clustering parameters. It runs clustering and returns a big table with the information
    regarding all the points in the ply being assigned to a cluster.
    """

    point_cloud_data_array = np.asarray(point_cloud_data_file.points)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(point_cloud_data_array)
    cluster_labels = dbscan.labels_
    unique_label_names = np.unique(cluster_labels)
    a = 0
    print(unique_label_names)
    for i in unique_label_names:
        for j in cluster_labels:
            if j == i:
                a += 1
        print("Cluster number: " + str(i))
        print("Number of points in the cluster: " + str(a))
        a = 0

    a = 0
    cluster_names_list = []
    cluster_count_list = []
    for i in unique_label_names:
        for j in cluster_labels:
            if j == i:
                a += 1
        if i != -1:
            cluster_names_list.append(i)
            cluster_count_list.append(a)
        a = 0
    clusters_df = pd.DataFrame({"Cluster_Number": cluster_names_list, "Cluster_Count": cluster_count_list})
    clusters_df = clusters_df.sort_values(by="Cluster_Count", ascending=False)
    # print(clusters_df.to_string())

    # Extract the vertices (points) and colors
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    # Filter points that are assigned to clusters (not -1)
    clustered_indices = np.where(cluster_labels != -1)[0]
    clustered_points = vertices[clustered_indices]
    original_colors = colors[clustered_indices]
    clustered_colors = cluster_labels

    return clustered_points, original_colors, clustered_colors, clusters_df


def plot_clusters(point_cloud_data_file, cluster_labels):
    """ This function provides a matplotlib 3d plot with a clustered point cloud"""

    # Assuming 'point_cloud_data' contains your 3D point cloud data as a NumPy array
    # Assuming 'cluster_labels' contains the cluster labels assigned by DBSCAN
    point_cloud_data = np.asarray(point_cloud_data_file.points)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get unique cluster labels, excluding -1 (noise label)
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])

    # Define a color map for visualization
    mapped_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    random.shuffle(mapped_colors)

    # Iterate through unique cluster labels and plot points
    for label, color in zip(unique_labels, mapped_colors):
        ax.scatter(point_cloud_data[cluster_labels == label][:, 0],
                   point_cloud_data[cluster_labels == label][:, 1],
                   point_cloud_data[cluster_labels == label][:, 2],
                   c=color, marker='o', s=5, label=f'Cluster {label}')

    # Customize plot labels and legend
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('DBSCAN Clustering of 3D Point Cloud')
    ax.legend()

    # Show the 3D plot
    plt.show()


def extract_camera_points(point_cloud_data_file):
    """ This function returns an array with points of only one color that represent the camera color"""
    # Creating arrays containing data about points and colors
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    # Define the color
    camera_color = [0, 1, 0]
    # Find indices of the dots with the defined color & then extract these points
    camera_indices = np.all(colors == camera_color, axis=1)
    camera_points = vertices[camera_indices]
    camera_points_colors = colors[camera_indices]
    return camera_points, camera_points_colors


def create_a_sphere(camera_points):
    central_point = np.mean(camera_points, axis=0)
    distances = pairwise_distances(camera_points, [central_point])
    # print(distances)
    distance_99 = np.percentile(distances, 99)
    distance = distance_99 * 1.0
    return [central_point, distance]


def create_a_cube(camera_points):
    # Find the minimum and maximum coordinates of the camera points
    min_coords = np.min(camera_points, axis=0)
    max_coords = np.max(camera_points, axis=0)

    # Calculate the side length of the cube
    factor = 0.99
    side_length = max(max_coords - min_coords) * factor

    # Calculate the central point of the cube
    central_point = (min_coords + max_coords) / 2

    return [central_point, side_length]


def crop_a_point_cloud(point_cloud_data_file, parameters: list, shape=['sphere', 'cube']):
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)

    if shape == 'sphere':
        # Extract the sphere parameters
        sphere_center, sphere_radius = parameters

        # Calculate the squared radius for efficient comparison
        sphere_radius_squared = sphere_radius ** 2

        # Calculate the squared distance from each point to the sphere center
        distances_squared = np.sum((vertices - sphere_center) ** 2, axis=1)

        # Find the indices of points inside the sphere
        indices_inside_sphere = np.where(distances_squared <= sphere_radius_squared)[0]

        # Extract the points and colors inside the sphere
        new_vertices = vertices[indices_inside_sphere]
        new_colors = colors[indices_inside_sphere]
    elif shape == 'cube':
        # Extract the cube parameters
        cube_center, side_length = parameters

        # Define the minimum and maximum coordinates of the cube
        min_coords = cube_center - side_length / 2
        max_coords = cube_center + side_length / 2

        # Find the indices of points inside the cube
        indices_inside_cube = np.all((vertices >= min_coords) & (vertices <= max_coords), axis=1)

        # Extract the points and colors inside the cube
        new_vertices = vertices[indices_inside_cube]
        new_colors = colors[indices_inside_cube]

    return new_vertices, new_colors


def calculate_rotation_and_scaling(circle_point_cloud):
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
        objective_function = lambda params: np.sum(
            (np.sqrt((x - params[0]) ** 2 + (y - params[1]) ** 2) - params[2]) ** 2)

        # Minimize the objective function to find the circle parameters
        result = minimize(objective_function, initial_guess, method='trust-constr')

        # Extract the fitted circle parameters
        circle_center = result.x[:2]
        circle_radius = result.x[2]

        return circle_center, circle_radius

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

    scaling_parameters = [rotation_matrix, scale_factor]

    return scaling_parameters





def transform_point_cloud(point_cloud, scaling_parameters):
    # Extract
    rotation_matrix, scale_factor = scaling_parameters
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


def green_index(point_cloud_data_file):
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    print(colors.shape)

    green_indices = []
    # Find indices of the dots with Green Index > 0.8
    for i in range(len(colors)):
        green_index_value = (colors[i, 1] - colors[i, 0]) / (colors[i, 1] + colors[i, 0])
        if green_index_value > -0.02:
            green_indices.append(i)
    filtered_points = vertices[green_indices]
    filtered_colors = colors[green_indices]
    return filtered_points, filtered_colors


def red_index(point_cloud_data_file):
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    print(colors.shape)

    red_indices = []
    # Find indices of the dots with Green Index > 0.8
    for i in range(len(colors)):
        red_index_value = (colors[i, 0] - colors[i, 1]) / (colors[i, 1] + colors[i, 0])
        if red_index_value > 0.25:
            red_indices.append(i)
    filtered_points = vertices[red_indices]
    filtered_colors = colors[red_indices]
    return filtered_points, filtered_colors


def create_alpha_shape(point_cloud_file_path, alpha, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    # Compute alpha shape
    alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        point_cloud, alpha=alpha
    )
    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path is not None:
        # Save alpha shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, alpha_shape)
    return alpha_shape


def create_poisson_shape(point_cloud_file_path, depth, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    # Compute Poisson shape
    poisson_shape, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=depth
    )

    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path:
        # Save Poisson shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, poisson_shape)

    # ChatGPT tells me that I can make sure everuthing is watertight right away. I haven't cheked it yet. If it's not working, I can use the calculate_watertight_volume function.
    # Ensure mesh is watertight
    poisson_shape.compute_vertex_normals()
    poisson_shape.remove_degenerate_triangles()
    poisson_shape.remove_unreferenced_vertices()

    return poisson_shape


def create_ball_pivoting_shape(point_cloud_file_path, radii, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    # Compute Ball pivoting shape
    ball_pivoting_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector(radii))

    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path:
        # Save Ball pivoting shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, ball_pivoting_shape)

    # Ensure mesh is watertight
    ball_pivoting_shape.compute_vertex_normals()
    ball_pivoting_shape.remove_degenerate_triangles()
    ball_pivoting_shape.remove_unreferenced_vertices()

    return ball_pivoting_shape


def create_convex_hull_shape(point_cloud_file_path, depth, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    # Compute convex_hull shape
    convex_hull_shape, _ = point_cloud.compute_convex_hull()

    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path:
        # Save convex_hull shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, convex_hull_shape)

    return convex_hull_shape


def calculate_watertight_volume(shape):
    vertices = np.asarray(shape.vertices)
    triangles = np.asarray(shape.triangles)
    # Create a Trimesh object
    # This library allows to make sure that each mesh is watertight
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    total_volume = 0
    for i in mesh.split():
        if i.is_watertight:
            total_volume += abs(i.volume)
    return total_volume


def calculate_shape_parameters(point_cloud_data_file, shape, total_volume):
    point_cloud_array = np.asarray(point_cloud_data_file.points)
    # Get dimensions (length, width, height)
    dimensions = np.ptp(point_cloud_array, axis=0)

    # Swap dimensions for height, length, and width
    dimensions = dimensions[[1, 2, 0]]

    # Calculate surface area
    surface_area = shape.get_surface_area()

    # Calculate aspect ratio
    aspect_ratio = np.max(dimensions) / np.min(dimensions)

    # Calculate elongation
    elongation = (np.max(dimensions) / np.median(dimensions)) - 1

    # Calculate flatness
    flatness = (np.min(dimensions) / np.median(dimensions)) - 1

    # Get connected components
    connected_components = shape.cluster_connected_triangles()

    # Initialize a dictionary to store parameters for each component
    component_parameters = {}

    # Calculate sphericity for the entire alpha shape
    sphericity = (np.pi ** (1 / 3)) * ((6 * total_volume) ** (2 / 3)) / surface_area

    # Calculate compactness for the entire alpha shape
    compactness = (36 * np.pi * total_volume ** 2) ** (1 / 3) / surface_area

    # Calculate number of independent components
    vertices = np.asarray(shape.vertices)
    triangles = np.asarray(shape.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    components = 0
    for i in mesh.split():
        components += 1

    # # Get points inside the alpha shape
    # points_inside = shape.select_by_index(np.arange(len(point_cloud)))

    # Get the number of points inside the alpha shape
    num_points_inside = len(point_cloud_array)
    point_density = num_points_inside / total_volume
    # Store parameters for the entire alpha shape
    parameters = {
        'Height': dimensions[0],
        'Length': dimensions[1],
        'Width': dimensions[2],
        'Volume': total_volume,
        'Surface_area': surface_area,
        'Aspect_ratio': aspect_ratio,
        'Elongation': elongation,
        'Flatness': flatness,
        'Sphericity': sphericity,
        'Compactness': compactness,
        'Components_number': components,
        'Point_density': point_density
    }
    return parameters


if __name__ == '__main__':
    folders_paths = paths.get_paths()

    # Set the runtype parameter
    runtype_options = [
        'whole',                # All files up to scaling not included
        'one_file',             # Same procedure for one file
        'red_from_clipped',     # Get red_from_clipped from clipped/cubes folder
        'scale',                # Run procedure of scaling using circles
        'create_alpha',         # Create alpha shapes and the csv based on corrected folder, get stats
    ]
    runtype = 'test'

    if runtype == 'test':
        for x, y in folders_paths.items():
            print(x, y)

    elif runtype == 'red_from_clipped':
        # Set import path
        clipped_folder_path = r'D:\results\plys\clipped\cubes'

        # Set export folders
        red_from_clipped_export_folder_path = r'D:\results\plys\clipped\cubes\red_from_clipped'

        plys = os.listdir(clipped_folder_path)
        for file in plys:
            ply_file_path = os.path.join(clipped_folder_path, file)
            if os.path.isfile(ply_file_path) and file != "complete_la_data.csv" and file != "set1_la_final.csv":
                # Set export paths
                red_from_clipped_export_file_path = os.path.join(red_from_clipped_export_folder_path, file)

                print(ply_file_path)
                print(red_from_clipped_export_file_path)

                # Open a file
                pcl = open_ply_file(ply_file_path)
                cropped_pcl = pcl

                # Color filtering
                vertices, colors = red_index(cropped_pcl)
                red_pcl = export_ply_file(vertices, colors, red_from_clipped_export_file_path)

    elif runtype == 'scale':
        circles_folder_path = r'D:\results\plys\circles'
        green_folder_path = r'D:\results\plys\clipped\cubes\clustered\color_filtered\green'
        rotated_export_folder_path = r'D:\results\plys\clipped\cubes\clustered\color_filtered\green\rotated'
        plys = os.listdir(circles_folder_path)
        for file in plys:
            ply_file_path = os.path.join(circles_folder_path, file)
            green_ply_file_path = os.path.join(green_folder_path, file)
            rotated_export_file_path = os.path.join(rotated_export_folder_path, file)
            if os.path.isfile(ply_file_path) and file != "complete_la_data.csv" and file != "set1_la_final.csv":
                print(ply_file_path)
                print(rotated_export_file_path)
                print(green_ply_file_path)

                pcl = open_ply_file(ply_file_path)
                circle_pcl = pcl
                green_pcl = open_ply_file(green_ply_file_path)

                scaling_parameters = calculate_rotation_and_scaling(circle_pcl)
                vertices, colors = transform_point_cloud(green_pcl, scaling_parameters)
                export_ply_file(vertices, colors, output_filepath=rotated_export_file_path)

    elif runtype == 'create_alpha':
        ply_folder_path = r'D:\results\plys\clipped\cubes\clustered\color_filtered\green\rotated\corrected'
        alpha_folder_path = r'D:\results\plys\clipped\cubes\clustered\color_filtered\green\rotated\alpha_shapes'
        csv_file_path = r'D:\results\plys\clipped\clustered\cubes\color_filtered\green\rotated\alpha_shapes\alpha_shapes1203.csv'

        plys = os.listdir(ply_folder_path)
        df = pd.DataFrame()

        for file in plys:
            ply_file_path = os.path.join(ply_folder_path, file)
            if os.path.isfile(ply_file_path):
                alpha_file_path = os.path.join(alpha_folder_path, file)
                print(ply_file_path)
                print(alpha_file_path)
                pcl = open_ply_file(ply_file_path)
                alpha_value = 0.5  # Adjust alpha as needed
                alpha_shape = create_alpha_shape(ply_file_path, alpha_file_path, alpha_value)
                total_volume = calculate_watertight_volume(alpha_shape)
                parameters = calculate_shape_parameters(pcl, alpha_shape, total_volume)
                parameters['File_name'] = file
                match = re.search(r'(\d+p\d+\.)', file)
                parameters['Measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
                df = pd.concat([df, pd.DataFrame([parameters])], ignore_index=True)
        print(df.to_string())
        df.to_csv(csv_file_path, index=False)

