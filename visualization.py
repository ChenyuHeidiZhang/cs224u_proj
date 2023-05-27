import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from constants import *
from cluster import find_dissimilarity_matrix

def visualize_dissimilarity_matrix(all_layer_repr):
    # TODO: N is too large, put them into bins
    dissimilarity = find_dissimilarity_matrix(all_layer_repr)
    plt.figure(figsize=(20, 20))
    plt.imshow(dissimilarity, cmap='hot', interpolation='nearest')
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.title("Dissimilarity matrix")
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"dissimilarity_matrix.png"), dpi=400)

def visualize_cluster_layer_neuron_count(cluster_id, num_clusters, distance_threshold, num_neurons_per_layer):
    dir = f"{VISUALIZATION_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/cluster_neuron_count"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.figure(figsize=(20, 10))
    plt.bar(range(NUM_LAYERS), num_neurons_per_layer)
    plt.xlabel("Layer")
    plt.ylabel("Number of neurons")
    plt.title(f"Cluster {cluster_id} number of neurons per layer")
    plt.savefig(os.path.join(dir, f"cluster_{cluster_id}.png"))

def visualize_cluster_scatter_plot(cluster_id, num_clusters, distance_threshold, layer_indices, neuron_indices):
    dir = f"{VISUALIZATION_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/cluster_scatter_plot"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.figure(figsize=(20, 10))
    plt.scatter(layer_indices, neuron_indices)
    plt.xlabel("Layer")
    plt.ylabel("Neuron index")
    plt.title(f"Cluster {cluster_id} scatter plot of neurons in each layer")
    plt.savefig(os.path.join(dir, f"cluster_{cluster_id}.png"))

def plot_cluster_neurons(cluster_labels, num_clusters, distance_threshold):
    # for cluster_id in range(max(cluster_labels)+1):
    for cluster_id in range(min(max(cluster_labels)+1, 50)):
        # find the indices of neurons in the the same cluster
        indices = np.where(cluster_labels == cluster_id)[0]
        # find the layer of each selected neuron
        layer_indices = indices // HIDDEN_DIM
        # find the neuron index within each layer
        neuron_indices = indices % HIDDEN_DIM
        # find number of selected neurons in each layer, there is a bin for each layer even if there is no neuron in that layer
        num_neurons_per_layer = np.zeros(NUM_LAYERS)
        for layer_id in range(NUM_LAYERS):
            num_neurons_per_layer[layer_id] = np.sum(layer_indices == layer_id)
        print(f"Cluser {cluster_id} number of neurons: {indices.shape[0]}")
        print(f"Cluser {cluster_id} number of neurons per layer: {num_neurons_per_layer}")
        # visualize_cluster_layer_neuron_count(cluster_id, num_clusters, distance_threshold, num_neurons_per_layer)
        visualize_cluster_scatter_plot(cluster_id, num_clusters, distance_threshold, layer_indices, neuron_indices)

def plot_cluster_top_tokens_neuron(cluster_id_to_top_token_indices, all_layer_repr, cluster_labels, num_clusters, distance_threshold, num_top_tokens):
    dir = f"{VISUALIZATION_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/cluster_top_tokens"
    if not os.path.exists(dir):
        os.makedirs(dir)
    # visualize a max of 50 clusters
    for cluster_id in range(min(max(cluster_labels)+1, 50)):
        top_tokens = cluster_id_to_top_token_indices[cluster_id]
        # plot a heatmap, where y axis is each token_id, x axis is each neuron, and the color is the activation of the neuron on the token
        neurons_all_tokens = []
        for token_id in top_tokens:
            # find the representation of the token
            neurons = all_layer_repr[:, token_id] # (N, 1)
            neurons_all_tokens.append(neurons)
        neurons_all_tokens = torch.stack(neurons_all_tokens, dim=1).T.numpy() # (num_top_tokens, N)
        # N is too large, put N neurons into 100 bin and calculate sum of each bin, so that the shape of neurons_all_tokens is a numpy array (num_top_tokens, 100)
        neurons_all_tokens = np.array([np.sum(neurons_all_tokens[:, i*100:(i+1)*100], axis=1) for i in range(100)]).T
        # # find indices of neurons in this cluster
        # indices = np.where(cluster_labels == cluster_id)[0]
        # # also plot indices of neurons in this cluster on the same plot and color them differently
        # neurons_all_tokens[:, indices] = 100
        plt.figure(figsize=(20, 10))
        plt.xlabel('Neuron in 100 bins')
        plt.ylabel('Token')
        plt.title(f'Cluster {cluster_id} top {num_top_tokens} tokens')
        plt.imshow(neurons_all_tokens, cmap='hot', interpolation='nearest')
        plt.savefig(os.path.join(dir, f"cluster_{cluster_id}_top_{num_top_tokens}_tokens.png"))

