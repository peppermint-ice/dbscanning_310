import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import db_clusterization
import get_camera_info
import open3d as o3d
import trimesh
import os
import re


def create_convex_hull(point_cloud, hull_output_path, combined_output_path):
    # Compute convex hull
    convex_hull, _ = point_cloud.compute_convex_hull(alpha=0.005)

    # Merge original point cloud and convex hull
    # Create a new PointCloud to store the result
    result_point_cloud = o3d.geometry.PointCloud()

    # Append the points from the original point cloud
    result_point_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)

    # Append the vertices from the convex hull
    result_point_cloud.points.extend(o3d.utility.Vector3dVector(convex_hull.vertices))

    # Save the result to a new .ply file
    output_file_path = combined_output_path
    o3d.io.write_point_cloud(output_file_path, result_point_cloud)
    o3d.io.write_triangle_mesh(hull_output_path, convex_hull)


def create_alpha_shape(input_file_path, alpha, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(input_file_path)

    # Compute alpha shape
    alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        point_cloud, alpha=alpha
    )
    if output_file_path:
        # Save alpha shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, alpha_shape)

    return alpha_shape


def calculate_watertight_volume(alpha_shape):
    vertices = np.asarray(alpha_shape.vertices)
    triangles = np.asarray(alpha_shape.triangles)
    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    total_volume = 0
    for i in mesh.split():
        if i.is_watertight:
            total_volume += abs(i.volume)
    return total_volume


def calculate_alpha_shape_parameters(point_cloud, alpha_shape, total_volume):
    # Get dimensions (length, width, height)
    dimensions = np.ptp(point_cloud, axis=0)

    # Swap dimensions for height, length, and width
    dimensions = dimensions[[1, 2, 0]]

    # # Project the alpha shape onto the xy-plane
    # projection_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # Identity matrix for xy-plane projection
    # projected_alpha_shape = alpha_shape.transform(projection_matrix)
    # # Extract the vertices of the projected alpha shape
    # projected_vertices = np.asarray(projected_alpha_shape.vertices)
    #
    # # Calculate the dimensions on the xy-plane
    # dimensions_xy = np.ptp(projected_vertices, axis=0)

    # # Get the largest and smallest dimensions
    # largest_dimension = np.max(dimensions_xy)
    # smallest_dimension = np.min(dimensions_xy)

    # Calculate surface area
    surface_area = alpha_shape.get_surface_area()

    # Calculate aspect ratio
    aspect_ratio = np.max(dimensions) / np.min(dimensions)

    # Calculate elongation
    elongation = (np.max(dimensions) / np.median(dimensions)) - 1

    # Calculate flatness
    flatness = (np.min(dimensions) / np.median(dimensions)) - 1

    # Get connected components
    connected_components = alpha_shape.cluster_connected_triangles()

    # Initialize a dictionary to store parameters for each component
    component_parameters = {}

    # Calculate sphericity for the entire alpha shape
    sphericity = (np.pi ** (1 / 3)) * ((6 * total_volume) ** (2 / 3)) / surface_area

    # Calculate compactness for the entire alpha shape
    compactness = (36 * np.pi * total_volume ** 2) ** (1 / 3) / surface_area

    # Calculate number of independent components
    vertices = np.asarray(alpha_shape.vertices)
    triangles = np.asarray(alpha_shape.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    components = 0
    for i in mesh.split():
        components += 1

    # # Get points inside the alpha shape
    # points_inside = alpha_shape.select_by_index(np.arange(len(point_cloud)))

    # Get the number of points inside the alpha shape
    num_points_inside = len(point_cloud)
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
    print(parameters)
    return parameters


if __name__ == '__main__':
    # ply_file_path = 'scaled_green.ply'
    # alpha_file_path = 'alpha_shape.ply'

    # point_cloud_array, point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
    # alpha_value = 0.5  # Adjust alpha as needed
    # alpha_shape = create_alpha_shape(ply_file_path, alpha_file_path, alpha_value)
    # total_volume = calculate_watertight_volume(alpha_shape)
    # calculate_alpha_shape_parameters(point_cloud_array, alpha_shape, total_volume)

    ply_folder_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\corrected'
    alpha_folder_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes'
    csv_file_path = r'D:\results\plys\clipped\clustered\color_filtered\green\rotated\alpha_shapes\df.csv'

    plys = os.listdir(ply_folder_path)
    df = pd.DataFrame()

    for file in plys:
        ply_file_path = os.path.join(ply_folder_path, file)
        if os.path.isfile(ply_file_path):
            alpha_file_path = os.path.join(alpha_folder_path, file)
            print(ply_file_path)
            print(alpha_file_path)
            point_cloud_array, point_cloud_file = db_clusterization.open_ply_file(ply_file_path)
            alpha_value = 0.5  # Adjust alpha as needed
            alpha_shape = create_alpha_shape(ply_file_path, alpha_file_path, alpha_value)
            total_volume = calculate_watertight_volume(alpha_shape)
            parameters = calculate_alpha_shape_parameters(point_cloud_array, alpha_shape, total_volume)
            parameters['File_name'] = file
            match = re.search(r'(\d+p\d+\.)', file)
            parameters['Measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
            df = pd.concat([df, pd.DataFrame([parameters])], ignore_index=True)
    print(df.to_string())
    df.to_csv(csv_file_path, index=False)
