import pandas as pd
import db_clusterization
import os
from matplotlib import pyplot as plt
import random

# Here, I'm trying to find optimal clustering results (number of clusters, sizes of the biggest clusters) for the plants
# I want to have a plant recognized as 1 or more clusters together with the pot. I also want to be able to identify
# these clusters. To start, I ran 25 plants with eps = 0.07 and min_samples = 80. After that, I've analyzed the results.
# As soon as I have good ranges for cluster sizes and number of clusters, I'll be able to automatically change eps and
# min_samples to achieve better results.

# This code is for whatever tasks I do while carrying out the analysis
# setting dbscanning parameters

# RESULTS after analyzing 50 images, I can clearly see that a good image usually has the biggest that contains
# 1/4 of all the point cloud.

eps = 0.09
min_samples = 70

# setting system directories for tests

ply_folder_path = r'D:\results\plys\selected'
export_folder_path = r'D:\results\plys\selected\clustered'
plys = os.listdir(ply_folder_path)

# this algorithm runs clustering for all files without saving the data. the information regarding all clusters and their
# sizes is saved into 'data' array

# data = []
# for file in plys:
#     df = pd.DataFrame()
#     ply_file_path = os.path.join(ply_folder_path, file)
#     if os.path.isfile(ply_file_path):
#         cluster_names_list = []
#         cluster_count_list = []
#         filenamestupidlist = []
#         export_file_path = os.path.join(export_folder_path, file)
#         print(ply_file_path)
#         point_cloud_array = db_clusterization.open_ply_file(ply_file_path)
#         cluster_names = db_clusterization.create_clusters(point_cloud_array, eps, min_samples)
#         unique_cluster_labels = db_clusterization.count_unique_cluster_labels(cluster_names)
#         a = 0
#         for i in unique_cluster_labels:
#             for j in cluster_names:
#                 if j == i:
#                     a += 1
#             cluster_names_list.append(i)
#             cluster_count_list.append(a)
#             filenamestupidlist.append(file)
#             a = 0
#         row = pd.DataFrame({"Cluster Number": cluster_names_list, "Cluster Count": cluster_count_list, "File Name": filenamestupidlist})
#         print(row)
#     df = pd.concat([df, row], ignore_index=True)
#     df = df.sort_values(by='Cluster Count', ascending=False)
#     data.append(df)
# print(len(data))
# print(data[0])
#
# # this algorithm creates a pivot table for all the cluster size information
#
# stats = pd.DataFrame({"Number of clusters": [],
#                         "The biggest cluster": [],
#                         'The second biggest cluster': [],
#                         'The third biggest cluster': [],
#                         "File Name": []})
#
# for dataframe in data:
#     print('File Name: ', dataframe['File Name'].unique()[0])
#     print('Number of clusters: ', len(dataframe))
#     print('The biggest cluster: ', dataframe['Cluster Count'].max())
#     print('The second biggest cluster: ',
#           dataframe["Cluster Count"].sort_values(ascending=False).iloc[1])
#     print('The third biggest cluster: ',
#           dataframe["Cluster Count"].sort_values(ascending=False).iloc[2])
#
#     row = pd.DataFrame({"Number of clusters": [len(dataframe)],
#                         "The biggest cluster": [dataframe['Cluster Count'].max()],
#                         'The second biggest cluster': [dataframe["Cluster Count"].sort_values(ascending=False).iloc[1]],
#                         'The third biggest cluster': [dataframe["Cluster Count"].sort_values(ascending=False).iloc[2]],
#                         "File Name": [dataframe['File Name'].unique()[0]]})
#     stats = pd.concat([stats, row], ignore_index=True)
# print(stats.to_string())


# just count all points in the image
point_cloud_sizes = []
file_names_list = []
for file in plys:
    df = pd.DataFrame()
    ply_file_path = os.path.join(ply_folder_path, file)
    if os.path.isfile(ply_file_path):
        export_file_path = os.path.join(export_folder_path, file)
        print(ply_file_path)
        point_cloud_array = db_clusterization.open_ply_file(ply_file_path)
        point_cloud_sizes.append(len(point_cloud_array))
        file_names_list.append(file)
df = pd.DataFrame({"File": file_names_list,
                   "Size": point_cloud_sizes})
print(df.to_string())
