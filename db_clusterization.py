""" This file is a set of functions that are needed to proceed with DB scanning algorithm. I'm using it as a library """
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plyfile
import random


def open_ply_file(file_path):
    """ This function opens a ply file and returns it as a numpy array """
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud_data_array = np.asarray(pcd.points)
    return point_cloud_data_array


def create_clusters(point_cloud_data, eps=0.15, min_samples=80):
    """ This function requires clustering parameters. It runs clustering and returns a big table with the information
    regarding all the points in the ply being assigned to a cluster """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(point_cloud_data)
    cluster_labels = dbscan.labels_
    return cluster_labels


def count_unique_cluster_labels(cluster_labels):
    """ This function gives statistics for every single cluster created in a file """
    unique_label_names = np.unique(cluster_labels)
    a = 0
    for i in unique_label_names:
        for j in cluster_labels:
            if j == i:
                a += 1
        print("Cluster number: " + str(i))
        print("Number of points in the cluster: " + str(a))
        a = 0
        print(unique_label_names)
    return unique_label_names


def count_all_cluster_sizes(cluster_labels, unique_label_names):
    """ This function returns a pd.DataFrame containing all clusters and their sizes for the file.
    There is NO -1 data in the returned dataframe"""
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
    print(clusters_df.to_string())
    return clusters_df


def plot_clusters_3d(point_cloud_data, cluster_labels):
    """ This function provides a matplotlig 3d plot with a clustered point cloud"""

    # Assuming 'point_cloud_data' contains your 3D point cloud data as a NumPy array
    # Assuming 'cluster_labels' contains the cluster labels assigned by DBSCAN

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get unique cluster labels, excluding -1 (noise label)
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])

    # Define a color map for visualization
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    random.shuffle(colors)

    # Iterate through unique cluster labels and plot points
    for label, color in zip(unique_labels, colors):
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


def export_clustered_data(point_cloud_data, cluster_labels, file_path):
    """ This function exports the clustered point cloud into a ply file """
    # Assuming 'point_cloud_data' contains your 3D point cloud data as a NumPy array
    # Assuming 'cluster_labels' contains the cluster labels assigned by DBSCAN

    # Get unique cluster labels, excluding -1 (noise label)
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])

    # Create a color mapping for each cluster
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    # Define a light grey color for noise points
    noise_color = [200, 200, 200]  # RGB values for light grey

    # Define the PLY file header (including 'red', 'green', 'blue' attributes)
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_data = np.zeros(len(point_cloud_data), dtype=vertex_dtype)
    vertex_data['x'] = point_cloud_data[:, 0]
    vertex_data['y'] = point_cloud_data[:, 1]
    vertex_data['z'] = point_cloud_data[:, 2]

    # Assign colors based on cluster labels, including noise points
    for label, color in zip(unique_labels, cluster_colors):
        cluster_mask = (cluster_labels == label)
        vertex_data['red'][cluster_mask] = int(color[0] * 255)
        vertex_data['green'][cluster_mask] = int(color[1] * 255)
        vertex_data['blue'][cluster_mask] = int(color[2] * 255)

    # Assign the light grey color to noise points (label -1)
    noise_mask = (cluster_labels == -1)
    vertex_data['red'][noise_mask] = noise_color[0]
    vertex_data['green'][noise_mask] = noise_color[1]
    vertex_data['blue'][noise_mask] = noise_color[2]

    # Create a PLY file
    ply = plyfile.PlyData([
        plyfile.PlyElement.describe(vertex_data, 'vertex', comments=['red', 'green', 'blue'])
    ])

    # Save the PLY file
    ply.write(file_path)


def run_dbscan(input_path, output_path, plot=False, eps=0.15, min_samples=80):
    """This function runs the whole workflow. Plotting is turned off by default """
    point_cloud_array = open_ply_file(input_path)
    cluster_names = create_clusters(point_cloud_array, eps, min_samples)
    unique_cluster_labels = count_unique_cluster_labels(cluster_names)
    stats = count_all_cluster_sizes(cluster_names, unique_cluster_labels)
    export_clustered_data(point_cloud_array, cluster_names, output_path)
    if plot:
        plot_clusters_3d(point_cloud_array, cluster_names)
    return stats


