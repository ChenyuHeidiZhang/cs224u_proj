from constants import *
import torch
import json
import os
import numpy as np


def load_neuron_repr():
    print('Loading neuron representations')
    neuron_representations_avg = {}
    for i in range(NUM_LAYERS):
        with open('neuron_repr/neuron_repr_{}.json'.format(i), 'r') as f:
            neuron_representations_avg[i] = torch.tensor(json.load(f)).t()  # shape (vocab_size, num_neurons) -> (num_neurons, vocab_size); num_neurons is hidden_dim

    # concatenate all layers
    all_layer_repr = torch.cat([neuron_representations_avg[i] for i in range(NUM_LAYERS)], dim=0) # (num_layers * num_neurons, vocab_size)
    return all_layer_repr

def save_cluster(cluster_labels, num_clusters, distance_threshold):
    dir = f'cluster_outputs/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    clusters = {}
    for cluster_id in range(max(cluster_labels)+1):
        # find the indices of neurons in the the same cluster
        indices = np.where(cluster_labels == cluster_id)[0]
        clusters[cluster_id] = indices.tolist()
    with open(os.path.join(dir, 'cluster_id_to_neurons.json'), 'w') as f:
        json.dump(clusters, f)

def load_cluster(num_clusters, distance_threshold):
    dir = f'cluster_outputs/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    with open(os.path.join(dir, 'cluster_id_to_neurons.json'), 'r') as f:
        clusters = json.load(f)
    return clusters