import json
import os
import torch

from constants import *
import utils

# Approach 1:
# cut out neurons that are far away from the cluster center, until the max distance to cluster center is less than threshold

# TODO: Approach 2:
# Define concepts for each cluster, and keep only neurons that activate for the concepts


def refine_clusters(num_clusters=50, max_distance_threshold=0.5):
    # load map from cluster id to list of neuron ids
    clusters = utils.load_cluster(num_clusters=num_clusters)

    # load neuron representations
    all_layer_repr = utils.load_neuron_repr(filtered=True)  # (all_num_neurons, vocab_size)
    # normalize each neuron's representation to be a unit vector
    all_layer_repr_norm = torch.nn.functional.normalize(all_layer_repr, p=2, dim=1)

    new_clusters = {}

    for cluster_idx, cluster in clusters.items():
        # set max distance to be a large number
        max_distance = 100
        cluster_neurons = cluster.copy()
        while max_distance > max_distance_threshold:
            print('num neurons in cluster {}: {}'.format(cluster_idx, len(cluster_neurons)))

            cluster_neuron_repr = all_layer_repr_norm[cluster_neurons]
            # compute cluster center
            cluster_center = torch.mean(cluster_neuron_repr, dim=0)
            # compute distance of each neuron to cluster center and find the neuron with max distance
            distances = torch.norm(cluster_neuron_repr - cluster_center, dim=1)
            max_distance, max_distance_idx = torch.max(distances, dim=0)
            print("Cluster {}: max distance to cluster center is {}".format(cluster_idx, max_distance))

            # remove the neuron with max distance from the cluster
            cluster_neurons.pop(max_distance_idx.item())

        new_clusters[cluster_idx] = cluster_neurons

    # save new clusters
    output_dir = CLUSTER_OUTPUT_DIR + '_refined'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{output_dir}/n_clusters{num_clusters}_max_dist_{max_distance_threshold}/clusters.json', 'w') as f:
        json.dump(new_clusters, f)


if __name__ == '__main__':
    refine_clusters(num_clusters=50, max_distance_threshold=0.5)
