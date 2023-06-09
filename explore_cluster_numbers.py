import os, json
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from utils import load_neuron_repr, load_and_mask_neuron_repr
from cluster import find_dissimilarity_matrix, find_tf_idf_neuron_repr
from constants import *


NUM_TOP_DISTANCE = 10
NUM_QUANTILE = 11
quantile_tensor = torch.arange(NUM_QUANTILE) / (NUM_QUANTILE - 1)



def cluster_internal_distance(dissimilarity_matrix, cluster_dict):
    # return a list where each element is the max distance within a cluster
    # cluster_dict: key is cluster_id, value is a list of neuron indices
    max_distances = []
    mean_distances = []
    std_distances = []
    lengths = []
    num_clusters = len(cluster_dict)
    quantile_list = []
    for cluster_id in range(num_clusters):
        cluster_size = len(cluster_dict[cluster_id])
        lengths.append(cluster_size)
        local_dissimilarity = dissimilarity_matrix[cluster_dict[cluster_id]][:, cluster_dict[cluster_id]]
        max_distances.append(torch.max(local_dissimilarity).numpy())
        lower_triangle_indices = torch.tril_indices(row=cluster_size,col=cluster_size,offset=-1).unbind()
        lower_triangle_values = local_dissimilarity[lower_triangle_indices]
        mean_distances.append(torch.mean(lower_triangle_values).numpy())
        std_distances.append(torch.std(lower_triangle_values).numpy())
        quantile_list.append(torch.quantile(lower_triangle_values, quantile_tensor))
    return lengths, max_distances, mean_distances, std_distances, quantile_list

def print_cluster_stats(dissimilarity_matrix, num_clusters_lst):
    for num_clusters in num_clusters_lst:
        print('num clusters:', num_clusters)
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete')
        cluster_labels = clustering.fit_predict(dissimilarity_matrix) # cluster label for each neuron
        clusters = {}
        for cluster_id in range(max(cluster_labels)+1):
            # find the indices of neurons in the the same cluster
            indices = np.where(cluster_labels == cluster_id)[0]
            clusters[cluster_id] = indices.tolist()
        length_list, max_distance_list, mean_distance_list, std_distance_list, quantile_list = cluster_internal_distance(dissimilarity_matrix, clusters)
        print(length_list)

        print("average mean: ", np.mean(mean_distance_list))
        print("average std: ", np.mean(std_distance_list))
        print("average max: ", np.mean(max_distance_list))
        print("----------------------------------")


def enumerate_cluster_number(dissimilarity_matrix, min_num_cluster, max_num_cluster, step, name_suffix=""):
    all_lengths_list = []
    all_max_distances_list = []
    all_mean_distances_list = []
    all_std_distances_list = []
    top_distances_list = []
    
    for num_clusters in range(min_num_cluster, max_num_cluster + 1, step):
        print('num clusters:', num_clusters)
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete')
        cluster_labels = clustering.fit_predict(dissimilarity_matrix) # cluster label for each neuron
        clusters = {}
        for cluster_id in range(max(cluster_labels)+1):
            # find the indices of neurons in the the same cluster
            indices = np.where(cluster_labels == cluster_id)[0]
            clusters[cluster_id] = indices.tolist()
        length_list, max_distance_list, mean_distance_list, std_distance_list, quantile_list = cluster_internal_distance(dissimilarity_matrix, clusters)
        all_max_distances_list.append(max_distance_list)
        all_mean_distances_list.append(mean_distance_list)
        all_std_distances_list.append(std_distance_list)
        all_lengths_list.append(length_list)
        top_distances = sorted(max_distance_list, reverse=True)[:NUM_TOP_DISTANCE]
        top_distances_list.append(top_distances)

        # plt.figure(figsize=(20, 20))
        # for cluster_idx in range(num_clusters):
        #     plt.scatter([cluster_idx] * NUM_QUANTILE, quantile_list[cluster_idx])
        # plt.xlabel("cluster index")
        # plt.ylabel("distances quantiles")
        # # plt.title(f"max dist vs. num clusters")
        # plt.savefig(os.path.join(VISUALIZATION_DIR, f"quantiles_of_{num_clusters}_clusters{name_suffix}.png"), dpi=400)


    # # plot all max distances versus num_clusters
    # plt.figure(figsize=(20, 20))
    # for num_clusters in range(min_num_cluster, max_num_cluster + 1, step):
    #     plt.scatter([num_clusters] * num_clusters, all_max_distances_list[(num_clusters-min_num_cluster) // step])
    # plt.xlabel("num clusters")
    # plt.ylabel("max distances")
    # plt.title(f"max dist vs. num clusters")
    # plt.savefig(os.path.join(VISUALIZATION_DIR, f"all_max_dist_vs_num_clusters_{min_num_cluster}-{max_num_cluster}{name_suffix}.png"), dpi=400)

    # # plot all mean distances versus num_clusters
    # plt.figure(figsize=(20, 20))
    # for num_clusters in range(min_num_cluster, max_num_cluster + 1, step):
    #     plt.scatter([num_clusters] * num_clusters, all_mean_distances_list[(num_clusters-min_num_cluster) // step])
    # plt.xlabel("num clusters")
    # plt.ylabel("mean distances")
    # plt.title(f"mean dist vs. num clusters")
    # plt.savefig(os.path.join(VISUALIZATION_DIR, f"all_mean_dist_vs_num_clusters_{min_num_cluster}-{max_num_cluster}{name_suffix}.png"), dpi=400)

    # # plot all std distances versus num_clusters
    # plt.figure(figsize=(20, 20))
    # for num_clusters in range(min_num_cluster, max_num_cluster + 1, step):
    #     plt.scatter([num_clusters] * num_clusters, all_std_distances_list[(num_clusters-min_num_cluster) // step])
    # plt.xlabel("num clusters")
    # plt.ylabel("std distances")
    # plt.title(f"std dist vs. num clusters")
    # plt.savefig(os.path.join(VISUALIZATION_DIR, f"all_std_dist_vs_num_clusters_{min_num_cluster}-{max_num_cluster}{name_suffix}.png"), dpi=400)

    # # plot all distances versus num_clusters with annotation
    # plt.figure(figsize=(30, 40))
    # for num_clusters in range(min_num_cluster, max_num_cluster + 1, step):
    #     plt.scatter([num_clusters] * num_clusters, all_max_distances_list[(num_clusters-min_num_cluster) // step])
    #     for idx, cluster_size in enumerate(all_lengths_list[(num_clusters-min_num_cluster) // step]):
    #         plt.annotate(cluster_size, (num_clusters+0.2, all_max_distances_list[(num_clusters-min_num_cluster) // step][idx]-0.001))
    # plt.xlabel("num clusters")
    # plt.ylabel("all distances")
    # plt.title(f"all dist vs. num clusters")
    # plt.savefig(os.path.join(VISUALIZATION_DIR, f"all_dist_vs_num_clusters_{min_num_cluster}-{max_num_cluster}_with_annotation{name_suffix}.png"), dpi=400)  

    return all_mean_distances_list



if __name__ == '__main__':
    # all_layer_repr = load_neuron_repr(filtered=True)
    # all_layer_repr = find_tf_idf_neuron_repr(all_layer_repr)
    # all_layer_repr = load_and_mask_neuron_repr(threshold=1.5)
    # dissimilarity = find_dissimilarity_matrix(all_layer_repr)
    # enumerate_cluster_number(dissimilarity, 10, 500, 10, name_suffix="_no_filtering")
    # print_cluster_stats(dissimilarity, range(10, 500, 10))


    
    # plot mean of mean distances versus num_clusters
    min_num_cluster = 10
    max_num_cluster = 500
    step = 10
    # no filtering
    all_layer_repr = load_neuron_repr(filtered=False)
    dissimilarity = find_dissimilarity_matrix(all_layer_repr)
    all_mean_distances_list_no_filtering = enumerate_cluster_number(dissimilarity, min_num_cluster, max_num_cluster, step, name_suffix="_no_filtering")
    mean_mean_distances_list_no_filtering = [np.mean(mean_list) for mean_list in all_mean_distances_list_no_filtering]
    # frequency filtering only
    all_layer_repr = load_neuron_repr(filtered=True)
    dissimilarity = find_dissimilarity_matrix(all_layer_repr)
    all_mean_distances_list_frequency_only = enumerate_cluster_number(dissimilarity, min_num_cluster, max_num_cluster, step, name_suffix="_frequency_only")
    mean_mean_distances_list_frequency_only = [np.mean(mean_list) for mean_list in all_mean_distances_list_frequency_only]
    # frequency filtering + smoothed
    with open(f"{NEURON_REPR_DIR}/neuron_repr_augmented.json", 'r') as f:
        all_layer_repr = torch.tensor(json.load(f))
    dissimilarity = find_dissimilarity_matrix(all_layer_repr)
    all_mean_distances_list_frequency_smoothed = enumerate_cluster_number(dissimilarity, min_num_cluster, max_num_cluster, step, name_suffix="_frequency_smoothed")
    mean_mean_distances_list_frequency_smoothed = [np.mean(mean_list) for mean_list in all_mean_distances_list_frequency_smoothed]
    plt.figure(figsize=(20, 20))
    plt.plot(range(min_num_cluster, max_num_cluster + 1, step), mean_mean_distances_list_no_filtering, label="no filtering")
    plt.plot(range(min_num_cluster, max_num_cluster + 1, step), mean_mean_distances_list_frequency_only, label="frequency_only")
    plt.plot(range(min_num_cluster, max_num_cluster + 1, step), mean_mean_distances_list_frequency_smoothed, label="frequency_smoothed")
    plt.xlabel("num clusters")
    plt.ylabel("mean of mean distances")
    plt.title(f"mean of mean dist vs. num clusters")
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"mean_of_mean_dist_vs_num_clusters_{min_num_cluster}-{max_num_cluster}.png"), dpi=400)
