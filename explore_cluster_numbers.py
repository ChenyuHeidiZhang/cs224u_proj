from utils import *
from visualization import *
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from constants import *


NUM_TOP_DISTANCE = 10


def cluster_internal_distance(dissimilarity_matrix, cluster_dict):
    # return a list where each element is the max distance within a cluster
    # cluster_dict: key is cluster_id, value is a list of neuron indices
    result = []
    num_clusters = len(cluster_dict)
    for cluster_id in range(num_clusters):
        local_dissimilarity = dissimilarity_matrix[cluster_dict[cluster_id]][:, cluster_dict[cluster_id]]
        result.append(torch.max(local_dissimilarity).numpy())
    return result
    

def enumerate_cluster_number(dissimilarity_matrix, min_num_cluster, max_num_cluster):
    all_distances_list = []
    top_distances_list = []
    
    for num_clusters in range(min_num_cluster, max_num_cluster + 1):
        print('num clusters:', num_clusters)
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete')
        cluster_labels = clustering.fit_predict(dissimilarity_matrix) # cluster label for each neuron
        clusters = {}
        for cluster_id in range(max(cluster_labels)+1):
            # find the indices of neurons in the the same cluster
            indices = np.where(cluster_labels == cluster_id)[0]
            clusters[cluster_id] = indices.tolist()
        max_distance_list = cluster_internal_distance(dissimilarity_matrix, clusters)
        all_distances_list.append(max_distance_list)
        top_distances = sorted(max_distance_list, reverse=True)[:NUM_TOP_DISTANCE]
        top_distances_list.append(top_distances)

    # plot all distances versus num_clusters
    plt.figure(figsize=(20, 20))
    for num_clusters in range(min_num_cluster, max_num_cluster + 1):
        plt.scatter([num_clusters] * num_clusters, all_distances_list[num_clusters-min_num_cluster])
    plt.xlabel("num clusters")
    plt.ylabel("all distances")
    plt.title(f"all dist vs. num clusters")
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"all_dist_vs_num_clusters_{min_num_cluster}-{max_num_cluster}.png"), dpi=400)

    # plot NUM_TOP_DISTANCE versus num_clusters
    top_distances_list = np.array(top_distances_list) # (max_num_cluster + 1 - min_num_cluster, NUM_TOP_DISTANCE)
    plt.figure(figsize=(20, 20))
    for idx in range(NUM_TOP_DISTANCE):
        plt.scatter(np.arange(min_num_cluster, max_num_cluster+1), top_distances_list[:, idx])
    plt.xlabel("num clusters")
    plt.ylabel("top distances")
    plt.title(f"top {NUM_TOP_DISTANCE} dist vs. num clusters")
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"top_{NUM_TOP_DISTANCE}_dist_vs_num_clusters_{min_num_cluster}-{max_num_cluster}.png"), dpi=400)
    


if __name__ == '__main__':
    all_layer_repr = load_neuron_repr()
    dissimilarity = find_dissimilarity_matrix(all_layer_repr)
    enumerate_cluster_number(dissimilarity, 10, 80)

