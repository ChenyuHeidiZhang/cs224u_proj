from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, BertModel, BertConfig, BertForPreTraining, BertForMaskedLM

from constants import *
import utils
from visualization import plot_cluster_neurons, plot_cluster_top_tokens_neuron, visualize_cluster_token_embeddings

def find_tf_idf_neuron_repr(all_layer_repr):
    # all_layer_repr is a tensor of shape N * D where N is the number of neurons and D is the dimensionality (vocab size)
    # Compute the term frequency (TF) matrix
    tf_matrix = all_layer_repr / torch.sum(all_layer_repr, dim=1, keepdim=True)

    # Compute the document frequency (DF) vector
    df_vector = torch.count_nonzero(tf_matrix, dim=0)

    # Compute the inverse document frequency (IDF) vector
    N = len(all_layer_repr)  # Total number of documents
    idf_vector = torch.log(torch.tensor(N, dtype=torch.float32) / (df_vector + 1))

    # Compute the TF-IDF matrix
    tfidf_matrix = tf_matrix  * idf_vector

    tfidf_matrix = torch.nan_to_num(tfidf_matrix, nan=0.0)

    return tfidf_matrix

def filter_less_popular_tokens(all_layer_repr, k=10000):
    """
    turn off tokens that are not commonly activated by neurons
    only keep top k
    """
    abs_all_layer_repr = torch.abs(all_layer_repr)
    # calculate the sum for each token across all neurons
    token_sum = torch.sum(abs_all_layer_repr, dim=0)
    # select the top k indices
    top_token_indices = torch.topk(token_sum, k=k)[1]
    # get indices that do not belong to top_token_indices
    non_top_token_indices = torch.nonzero(torch.sum(top_token_indices.unsqueeze(1) == torch.arange(VOCAB_SIZE).unsqueeze(0), dim=0) == 0).squeeze(1)
    # set to 0
    all_layer_repr[:, non_top_token_indices] = 0
    return all_layer_repr

def find_dissimilarity_matrix(all_layer_repr, similarity_method= 'cosine'):
    # Normalize the input tensor
    print('Normalizing input tensor')
    # normalize each neuron's representation to be a unit vector
    input1_norm = torch.nn.functional.normalize(all_layer_repr, p=2, dim=1)
    input2_norm = input1_norm.clone()

    if similarity_method == 'cosine':
        # Compute the cosine similarity using matrix multiplication
        # similarity is now a matrix of shape (N, N) containing pairwise cosine similarities
        print('Computing cosine similarity')
        similarity = torch.mm(input1_norm, input2_norm.t())
        print('Done computing similarity matrix. Shape:', similarity.shape)
    elif similarity_method == 'pearson':
        # compute the pearson correlation coefficient 
        # the result should be a N * N matrix
        similarity = torch.corrcoef(all_layer_repr)
    else:
        raise ValueError(f"Similarity method {similarity_method} is not supported")

    # Convert similarity to dissimilarity matrix
    dissimilarity = 1 - similarity
    return dissimilarity


def get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens, token_filtered=True):
    dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    # dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_threshold1/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    if token_filtered:
        with open(f'{NEURON_REPR_DIR}/token_ids_to_keep.json', 'r') as f:
            indices_to_token_ids = json.load(f)

    cluster_id_to_top_token_indices = {}
    # Find top-k tokens that are activated by neurons in the same cluster and write to a file
    with open(os.path.join(dir, f"top_{num_top_tokens}_tokens.txt"), 'w') as f:
        for cluster_id in range(max(cluster_labels)+1):
            # find the indices of neurons in the the same cluster
            indices = np.where(cluster_labels == cluster_id)[0]
            # print(indices)
            # aggregate activations of neurons in the cluster
            # TODO: check if abs() is better
            cluster_activations = torch.sum(all_layer_repr[indices], dim=0) # D
            # find the indices of the top-k tokens
            top_k_indices = torch.topk(cluster_activations, k=num_top_tokens)[1]
            # convert indices to tokens
            if token_filtered:
                top_k_indices = [indices_to_token_ids[i] for i in top_k_indices]
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
            print("Cluster {}: {}".format(cluster_id, top_k_tokens))
            f.write("Cluster {}: {}\n".format(cluster_id, top_k_tokens))
            cluster_id_to_top_token_indices[cluster_id] = top_k_indices
    return cluster_id_to_top_token_indices


def compute_clusters(all_layer_repr, tokenizer, num_clusters=3, distance_threshold=None, num_top_tokens=10, token_filtered=True):
    # input tensors all_layer_repr is of shape (N, D)
    # where N is the numbers of neurons (num_layers * num_neurons_per_layer = 9984), and D is the dimensionality (vocab size)

    # Compute the dissimilarity matrix
    dissimilarity = find_dissimilarity_matrix(all_layer_repr)

    # Apply Agglomerative Hierarchical Clustering
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete', distance_threshold=distance_threshold)
    cluster_labels = clustering.fit_predict(dissimilarity) # cluster label for each neuron

    # Save the cluster labels
    utils.save_cluster(cluster_labels, num_clusters, distance_threshold)
    # print(cluster_labels)

    # Find top-k tokens that are activated by neurons in the same cluster and write to a file
    cluster_id_to_top_token_indices = get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens, token_filtered=token_filtered)

    # plot the positions (layer and index) of neurons for each cluster label
    # plot_cluster_neurons(cluster_labels, num_clusters, distance_threshold)

    # TODO: this should really be done across clusters, not within clusters, to show that the clusters are well-separated
    # plot the top tokens for each cluster with their representations
    # plot_cluster_top_tokens_neuron(cluster_id_to_top_token_indices, all_layer_repr, cluster_labels, num_clusters, distance_threshold, num_top_tokens)


def explore_cluster_distance_thresholds(dissimilarity, thresholds):
    for threshold in thresholds:
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=threshold)
        cluster_labels = clustering.fit_predict(dissimilarity) # cluster label for each neuron
        cluster_size = max(cluster_labels) + 1
        print(f"Threshold: {threshold}, Number of clusters: {cluster_size}")


def run():
    all_layer_repr = utils.load_neuron_repr(filtered=True)
    # all_layer_repr = filter_less_popular_tokens(all_layer_repr, k=10000)
    # all_layer_repr = utils.load_augmented_neuron_repr()
    # all_layer_repr = find_tf_idf_neuron_repr(all_layer_repr)
    # replace nan with 0
    # all_layer_repr = torch.nan_to_num(all_layer_repr)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # one of num_cluster and distance_threshold must be None
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=20, distance_threshold=None, num_top_tokens=10)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=50, distance_threshold=None, num_top_tokens=30, token_filtered=True)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=500, distance_threshold=None, num_top_tokens=10)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=None, distance_threshold=0.999, num_top_tokens=10)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=50, distance_threshold=None, num_top_tokens=30, token_filtered=True)
    compute_clusters(all_layer_repr, tokenizer, num_clusters=200, distance_threshold=None, num_top_tokens=30, token_filtered=True)


if __name__ == '__main__':
    run()

    # visualize_cluster_token_embeddings(folder_name="n_clusters50_distance_threshold_None", num_tokens_per_cluster=30)
