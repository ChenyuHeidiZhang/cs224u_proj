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

def find_dissimilarity_matrix(all_layer_repr):
    # Normalize the input tensor
    print('Normalizing input tensor')
    # normalize each neuron's representation to be a unit vector
    input1_norm = torch.nn.functional.normalize(all_layer_repr, p=2, dim=1)
    input2_norm = input1_norm.clone()

    # Compute the cosine similarity using matrix multiplication
    # similarity is now a matrix of shape (N, N) containing pairwise cosine similarities
    print('Computing cosine similarity')
    similarity = torch.mm(input1_norm, input2_norm.t()).abs()
    print('Done computing similarity matrix. Shape:', similarity.shape)

    # Convert similarity to dissimilarity matrix
    dissimilarity = 1 - similarity
    return dissimilarity


def get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens):
    dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    # dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_threshold1/'
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


def compute_clusters(all_layer_repr, tokenizer, num_clusters=3, distance_threshold=None, num_top_tokens=10):
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
    cluster_id_to_top_token_indices = get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens)

    # plot the positions (layer and index) of neurons for each cluster label
    plot_cluster_neurons(cluster_labels, num_clusters, distance_threshold)

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
    all_layer_repr = utils.load_neuron_repr()
    # all_layer_repr = utils.load_and_mask_neuron_repr(threshold=1)
    all_layer_repr = find_tf_idf_neuron_repr(all_layer_repr)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # one of num_cluster and distance_threshold must be None
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=20, distance_threshold=None, num_top_tokens=10)
    compute_clusters(all_layer_repr, tokenizer, num_clusters=50, distance_threshold=None, num_top_tokens=10)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=500, distance_threshold=None, num_top_tokens=10)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=None, distance_threshold=0.999, num_top_tokens=10)


if __name__ == '__main__':
    run()

    # visualize_cluster_token_embeddings(folder_name="n_clusters50_distance_threshold_None")
