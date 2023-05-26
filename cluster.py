from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, BertModel, BertConfig

NUM_LAYERS = 13

def load_neuron_repr():
    print('Loading neuron representations')
    neuron_representations_avg = {}
    for i in range(NUM_LAYERS):
        with open('neuron_repr/neuron_repr_{}.json'.format(i), 'r') as f:
            neuron_representations_avg[i] = torch.tensor(json.load(f)).t()  # shape (vocab_size, num_neurons) -> (num_neurons, vocab_size); num_neurons is hidden_dim

    # concatenate all layers
    all_layer_repr = torch.cat([neuron_representations_avg[i] for i in range(NUM_LAYERS)], dim=0) # (num_layers * num_neurons, vocab_size)
    return all_layer_repr


def visualize_cluster(cluster_id, num_neurons_per_layer):
    plt.figure(figsize=(20, 10))
    plt.bar(range(13), num_neurons_per_layer)
    plt.xlabel("Layer")
    plt.ylabel("Number of neurons")
    plt.title(f"Cluster {cluster_id} number of neurons per layer")
    plt.savefig(f"visualizations/cluster_neuron_count/cluster_{cluster_id}.png")


def compute_clusters(all_layer_repr, tokenizer, num_clusters=3, num_top_tokens=10):
    # input tensors all_layer_repr is of shape (N, D)
    # where N is the numbers of neurons (num_layers * num_neurons_per_layer = 9984), and D is the dimensionality (vocab size)

    # Normalize the input tensor
    print('Normalizing input tensor')
    input1_norm = torch.nn.functional.normalize(all_layer_repr, p=2, dim=1)
    input2_norm = input1_norm.clone()

    # Compute the cosine similarity using matrix multiplication
    # similarity is now a matrix of shape (N, N) containing pairwise cosine similarities
    print('Computing cosine similarity')
    similarity = torch.mm(input1_norm, input2_norm.t())
    print('Done computing similarity matrix. Shape:', similarity.shape)

    # Convert similarity to dissimilarity matrix
    dissimilarity = 1 - similarity

    # Apply Agglomerative Hierarchical Clustering
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete', distance_threshold=None)
    cluster_labels = clustering.fit_predict(dissimilarity) # cluster label for each neuron

    # Print the cluster labels
    print(cluster_labels)

    # Find top-k tokens that are activated by neurons in the same cluster
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

    # plot the neurons with their cluster labels
    for cluster_id in range(max(cluster_labels)+1):
        # find the indices of neurons in the the same cluster
        indices = np.where(cluster_labels == cluster_id)[0]
        # find the layer of each selected neuron
        layer_indices = indices // 768
        # find the neuron index within each layer
        neuron_indices = indices % 768
        # find number of selected neurons in each layer
        num_neurons_per_layer = np.bincount(layer_indices)
        print(f"Cluser {cluster_id} number of neurons: {indices.shape[0]}")
        print(f"Cluser {cluster_id} number of neurons per layer: {num_neurons_per_layer}")
        visualize_cluster(cluster_id, num_neurons_per_layer)
    
    # TODO: plot the top tokens for each cluster with their representations


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
    compute_clusters(all_layer_repr, tokenizer, num_clusters=20, num_top_tokens=10)


if __name__ == '__main__':
    # test()
    run()
