import json
import os
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from constants import *
import utils

# Approach 1:
# cut out neurons that are far away from the cluster center, until the max distance to cluster center is less than threshold

# TODO: Approach 2:
# Define concepts for each cluster, and keep only neurons that activate for the concepts

REFINED_DIR = CLUSTER_OUTPUT_DIR + '_refined'

def refine_clusters(num_clusters=50, max_distance_threshold=0.6, topk_tokens=10, token_filtered=True):
    # load map from cluster id to list of neuron ids
    clusters = utils.load_cluster(num_clusters=num_clusters)

    # load neuron representations; use original filtered representation even if augmented representation is used for clustering
    all_layer_repr = utils.load_neuron_repr(filtered=token_filtered)  # (all_num_neurons, vocab_size)
    # normalize each neuron's representation to be a unit vector
    all_layer_repr_norm = torch.nn.functional.normalize(all_layer_repr, p=2, dim=1)

    new_clusters = {}
    all_distance_history = []

    for cluster_idx, cluster in clusters.items():
        # set max distance to be a large number
        max_distance = 100
        cluster_neurons = cluster.copy()
        print('num neurons in cluster {}: {}'.format(cluster_idx, len(cluster_neurons)))
        distance_history = []
        while max_distance > max_distance_threshold:
            cluster_neuron_repr = all_layer_repr_norm[cluster_neurons]
            # compute cluster center
            cluster_center = torch.mean(cluster_neuron_repr, dim=0)
            # compute distance of each neuron to cluster center and find the neuron with max distance
            distances = torch.norm(cluster_neuron_repr - cluster_center, dim=1)
            max_distance, max_distance_idx = torch.max(distances, dim=0)
            distance_history.append(max_distance.item())

            # remove the neuron with max distance from the cluster
            cluster_neurons.pop(max_distance_idx.item())
        print('remaining num neurons in cluster {}: {}'.format(cluster_idx, len(cluster_neurons)))
        print("Cluster {}: max distance to cluster center is {}".format(cluster_idx, max_distance))
        print(distance_history)
        all_distance_history.append(distance_history + [len(cluster)])

        new_clusters[cluster_idx] = cluster_neurons

    # save distance history
    with open(f'{VISUALIZATION_DIR}/n_clusters{num_clusters}_distance_threshold_None/refinement_distance_history.json', 'w') as f:
        json.dump(all_distance_history, f)

    # # save new clusters
    # out_path = f'{REFINED_DIR}/n_clusters{num_clusters}_max_dist_{max_distance_threshold}'
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # with open(f'{out_path}/cluster_id_to_neurons.json', 'w') as f:
    #     json.dump(new_clusters, f)

    # compute topk tokens for each new cluster
    # compute_topk_tokens(new_clusters, all_layer_repr_norm, num_clusters=num_clusters, max_distance_threshold=max_distance_threshold, topk_tokens=topk_tokens, token_filtered=token_filtered)


def plot_distance_history(num_clusters=50):
    with open(f'{VISUALIZATION_DIR}/n_clusters{num_clusters}_distance_threshold_None/refinement_distance_history.json', 'r') as f:
        all_distance_history = json.load(f)

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    # plot distance history for 8 random clusters
    # the last element in each list is the number of neurons in the cluster
    random_cluster_ids = np.random.choice(np.arange(num_clusters), size=8, replace=False)
    for i, cluster_id in enumerate(random_cluster_ids):
        num_neurons = all_distance_history[cluster_id].pop()
        plt.plot(np.arange(len(all_distance_history[cluster_id])), all_distance_history[cluster_id], label='c {}: {} neurons'.format(cluster_id, num_neurons))
    plt.xlabel('iteration')
    plt.ylabel('max distance to cluster center')
    plt.legend()
    plt.savefig(f'{VISUALIZATION_DIR}/n_clusters{num_clusters}_distance_threshold_None/refinement_distance_history.png')


def compute_topk_tokens(new_clusters, all_layer_repr_norm, num_clusters=50, max_distance_threshold=0.5, topk_tokens=10, token_filtered=True):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if token_filtered:
        with open(f'{NEURON_REPR_DIR}/token_ids_to_keep.json', 'r') as f:
            indices_to_token_ids = json.load(f)

    # compute topk tokens for each new cluster
    print('Computing topk tokens for each new cluster...')
    tokens_output_fn = f'{REFINED_DIR}/n_clusters{num_clusters}_max_dist_{max_distance_threshold}/top_{topk_tokens}_tokens.txt'
    with open(tokens_output_fn, 'w') as f:
        for cluster_id, cluster in new_clusters.items():
            # load neuron representations
            all_layer_repr = all_layer_repr_norm[cluster]
            # as the cluster activation, sum up activations of all neurons in the cluster
            cluster_activations = torch.sum(all_layer_repr, dim=0)
            # find topk tokens
            topk_activations, top_k_indices = torch.topk(cluster_activations, k=topk_tokens)
            # convert indices to token ids
            if token_filtered:
                top_k_indices = [indices_to_token_ids[i] for i in top_k_indices]
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
            print("Cluster {}: {}".format(cluster_id, top_k_tokens))
            f.write("Cluster {}: {}\n".format(cluster_id, top_k_tokens))



if __name__ == '__main__':
    # refine_clusters(num_clusters=50, max_distance_threshold=0, topk_tokens=30)
    plot_distance_history(num_clusters=50)