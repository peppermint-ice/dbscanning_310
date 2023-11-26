import pandas as pd
from db_clusterization import run_dbscan
import db_clusterization
import os
import get_camera_info

# Set the parameters

# eps = 0.09
# min_samples = 70
#
# # To run for 1 file
#
# ply_file_path = 'color_filtered_red.ply'
# exported_file_path = 'pot_clustered.ply'
# clusters = run_dbscan(
#     ply_file_path,
#     exported_file_path,
#     original_colors=True,
#     eps=eps,
#     min_samples=min_samples,
#     plot=True
# )

# To run for the whole folder

# ply_folder_path = r'D:\results\plys\selected'
# export_folder_path = r'D:\results\plys\selected\clustered'
# plys = os.listdir(ply_folder_path)
#
# for file in plys:
#     ply_file_path = os.path.join(ply_folder_path, file)
#     if os.path.isfile(ply_file_path):
#         export_file_path = os.path.join(export_folder_path, file)
#         print(ply_file_path)
#         print(export_file_path)
#         clusters = run_dbscan(ply_file_path, export_file_path, eps=eps, min_samples=min_samples)


# To run it so that every point cloud gets the biggest cluster over 30000 points

ply_folder_path = r'D:\results\plys\clipped'
export_folder_path = r'D:\results\plys\clipped\clustered'
plys = os.listdir(ply_folder_path)
eps_list = []
min_samples_list = []
file_names_list = []

for file in plys:
    eps = 0.09  # setting starting eps and min samples
    min_samples = 70
    ply_file_path = os.path.join(ply_folder_path, file)
    if os.path.isfile(ply_file_path):   # i might have some folders in the dir :)
        export_file_path = os.path.join(export_folder_path, file)
        print("Original path: ", ply_file_path)
        print("Clustered path: ", export_file_path)
#         # running the procedure
#         point_cloud_array = db_clusterization.open_ply_file(ply_file_path)
#         cluster_names = db_clusterization.create_clusters(point_cloud_array, eps, min_samples)
#         unique_cluster_labels = db_clusterization.count_unique_cluster_labels(cluster_names)
#         clusters_dataframe = db_clusterization.count_all_cluster_sizes(cluster_names, unique_cluster_labels)
#         while clusters_dataframe["Cluster_Count"].max() / len(point_cloud_array) < 0.25:    # changing parameters if the clusters are bad
#             eps += 0.005
#             # running the procedure until it fits the conditions
#             cluster_names = db_clusterization.create_clusters(point_cloud_array, eps, min_samples)
#             unique_cluster_labels = db_clusterization.count_unique_cluster_labels(cluster_names)
#             clusters_dataframe = db_clusterization.count_all_cluster_sizes(cluster_names, unique_cluster_labels)
#         # just for statistics, I want to have all epsilons and min_samples stored
#         eps_list.append(eps)
#         file_names_list.append(file)
#         # actually export the data
#         db_clusterization.export_clustered_data(point_cloud_array, cluster_names, export_file_path)
# # print out the stats
# stats = pd.DataFrame({"Epsilons": eps_list, "File Name": file_names_list})
# print(stats.to_string())
