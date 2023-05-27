from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, BertModel, BertConfig

NUM_LAYERS = 13
BATCH_SIZE = 64
HIDDEN_DIM = 768

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

def visualize_cluster_layer_neuron_count(cluster_id, num_clusters, distance_threshold, num_neurons_per_layer):
    dir = f"visualizations/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/cluster_neuron_count"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.figure(figsize=(20, 10))
    plt.bar(range(NUM_LAYERS), num_neurons_per_layer)
    plt.xlabel("Layer")
    plt.ylabel("Number of neurons")
    plt.title(f"Cluster {cluster_id} number of neurons per layer")
    plt.savefig(os.path.join(dir, f"cluster_{cluster_id}.png"))

def visualize_cluster_scatter_plot(cluster_id, num_clusters, distance_threshold, layer_indices, neuron_indices):
    dir = f"visualizations/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/cluster_scatter_plot"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.figure(figsize=(20, 10))
    plt.scatter(layer_indices, neuron_indices)
    plt.xlabel("Layer")
    plt.ylabel("Neuron index")
    plt.title(f"Cluster {cluster_id} scatter plot of neurons in each layer")
    plt.savefig(os.path.join(dir, f"cluster_{cluster_id}.png"))

def find_dissimilarity_matrix(all_layer_repr):
    # Normalize the input tensor
    print('Normalizing input tensor')
    # normalize each neuron's representation to be a unit vector
    input1_norm = torch.nn.functional.normalize(all_layer_repr, p=2, dim=1)
    input2_norm = input1_norm.clone()

    # Compute the cosine similarity using matrix multiplication
    # similarity is now a matrix of shape (N, N) containing pairwise cosine similarities
    print('Computing cosine similarity')
    similarity = torch.mm(input1_norm, input2_norm.t())
    print('Done computing similarity matrix. Shape:', similarity.shape)

    # Convert similarity to dissimilarity matrix
    dissimilarity = 1 - similarity
    return dissimilarity

def explore_cluster_distance_thresholds(dissimilarity, thresholds):
    for threshold in thresholds:
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=threshold)
        cluster_labels = clustering.fit_predict(dissimilarity) # cluster label for each neuron
        cluster_size = max(cluster_labels) + 1
        print(f"Threshold: {threshold}, Number of clusters: {cluster_size}")

def get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens):
    dir = f'cluster_outputs/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    cluster_id_to_top_token_indices = {}
    # Find top-k tokens that are activated by neurons in the same cluster and write to a file
    with open(os.path.join(dir, f"top_{num_top_tokens}_tokens.txt"), 'w') as f:
        for cluster_id in range(max(cluster_labels)+1):
            # find the indices of neurons in the the same cluster
            indices = np.where(cluster_labels == cluster_id)[0]
            # print(indices)
            # aggregate activations of neurons in the cluster
            cluster_activations = torch.sum(all_layer_repr[indices], dim=0) # D
            # find the indices of the top-k tokens
            top_k_indices = torch.topk(cluster_activations, k=num_top_tokens)[1]
            # convert indices to tokens
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
            print("Cluster {}: {}".format(cluster_id, top_k_tokens))
            f.write("Cluster {}: {}\n".format(cluster_id, top_k_tokens))
            cluster_id_to_top_token_indices[cluster_id] = top_k_indices
    return cluster_id_to_top_token_indices

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
    dir = f"visualizations/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/cluster_top_tokens"
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

def compute_clusters(all_layer_repr, tokenizer, num_clusters=3, distance_threshold=None, num_top_tokens=10):
    # input tensors all_layer_repr is of shape (N, D)
    # where N is the numbers of neurons (num_layers * num_neurons_per_layer = 9984), and D is the dimensionality (vocab size)

    # Compute the dissimilarity matrix
    dissimilarity = find_dissimilarity_matrix(all_layer_repr)

    # Apply Agglomerative Hierarchical Clustering
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete', distance_threshold=distance_threshold)
    cluster_labels = clustering.fit_predict(dissimilarity) # cluster label for each neuron

    # Save the cluster labels
    save_cluster(cluster_labels, num_clusters, distance_threshold)

    # Print the cluster labels
    print(cluster_labels)

    # Find top-k tokens that are activated by neurons in the same cluster and write to a file
    cluster_id_to_top_token_indices = get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens)

    # plot the neurons with their cluster labels
    plot_cluster_neurons(cluster_labels, num_clusters, distance_threshold)
    
    # plot the top tokens for each cluster with their representations
    plot_cluster_top_tokens_neuron(cluster_id_to_top_token_indices, all_layer_repr, cluster_labels, num_clusters, distance_threshold, num_top_tokens)
    

def test():
    repr = torch.tensor([[1.4, 0], [0, 11], [0, 1]])  # first neuron activates on the 1st token, second neuron activates on the 2nd & 3rd token
    repr1 = torch.tensor([[1, 0], [0, 11], [0, 2]])  # first neuron activates on the 1st token, second neuron activates on the 2nd & 3rd token
    all_layer_repr = {0: repr.t(), 1: repr1.t()}
    # combine all layers
    all_layer_repr = torch.cat([all_layer_repr[i] for i in range(2)], dim=0)
    print(all_layer_repr.shape)  # (4, 3)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    compute_clusters(all_layer_repr, tokenizer, num_clusters=2, num_top_tokens=2)


def run():
    all_layer_repr = load_neuron_repr()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # one of num_cluster and distance_threshold must be None
    compute_clusters(all_layer_repr, tokenizer, num_clusters=20, distance_threshold=None, num_top_tokens=10)
    compute_clusters(all_layer_repr, tokenizer, num_clusters=500, distance_threshold=None, num_top_tokens=10)
    compute_clusters(all_layer_repr, tokenizer, num_clusters=None, distance_threshold=0.999, num_top_tokens=10)


if __name__ == '__main__':
    # test()
    run()
