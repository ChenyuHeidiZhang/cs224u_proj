import os
import json
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

import fasttext.util
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from constants import *

def visualize_dissimilarity_matrix(dissimilarity):
    # TODO: N is too large, put them into bins
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


def visualize_cluster_token_embeddings(folder_name, max_clusters_to_plot=5, num_tokens_per_cluster=10):
    # Load a FastText model
    # Note: You can download a pre-trained FastText model from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/
    fasttext.util.download_model('en', if_exists='ignore')  # English
    print("Loading FastText model...")
    ft = fasttext.load_model('cc.en.300.bin')

    plt.figure(figsize=(10, 10))

    tokens_file = os.path.join(CLUSTER_OUTPUT_DIR, folder_name, f"top_{num_tokens_per_cluster}_tokens.txt")
    all_tokens = []
    embeddings_all = []
    distances_per_cluster = []
    print('Computing embeddings...')
    with open(tokens_file, 'r') as f:
        for id, line in tqdm(enumerate(f)):
            # if id > max_clusters_to_plot: break
            tokens = line.split(': [')[-1].split('\']')[0].split(', ')
            tokens = [token.strip("'") for token in tokens]
            all_tokens.append(tokens)
            embeddings = np.array([ft.get_word_vector(token) for token in tokens])
            center_embedding = np.mean(embeddings, axis=0)
            # compute the distance of each token embedding to the center embedding
            distances = np.linalg.norm(embeddings - center_embedding, axis=1)
            # accumulate average distance of each cluster
            distances_per_cluster.append(np.mean(distances))
            embeddings_all.append(embeddings)

    # sort the clusters by average distance
    sorted_cluster_ids = np.argsort(distances_per_cluster)  # (num_clusters, )
    # sorted_cluster_ids = np.arange(len(all_tokens))
    # concate embeddings of clusters with the smallest average distance
    embeddings_all = np.concatenate([embeddings_all[i] for i in sorted_cluster_ids[:max_clusters_to_plot]], axis=0)
    # embeddings_all = np.concatenate(embeddings_all, axis=0)

    # tsne = TSNE(n_components=2)
    # embeddings_2d = tsne.fit_transform(embeddings_all)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_all)

    # print(embeddings_2d.shape)  # (num_tokens, 2)

    print('Plotting...')
    mapname = 'rainbow' if max_clusters_to_plot < 10 else 'tab20'
    color_map = plt.cm.get_cmap(mapname, max_clusters_to_plot)
    for i, cluster_id in enumerate(sorted_cluster_ids[:max_clusters_to_plot]):
        start_idx = i * num_tokens_per_cluster
        xys = embeddings_2d[start_idx:start_idx+num_tokens_per_cluster, :]
        xs, ys = xys[:, 0], xys[:, 1]
        # print(xs, ys)
        plt.scatter(xs, ys, color=color_map(i), label=f'cluster_id {cluster_id}')
        if max_clusters_to_plot < 5:
            for j in range(num_tokens_per_cluster):
                x, y = xs[j], ys[j]
                token = all_tokens[cluster_id][j]
                plt.annotate(token, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATION_DIR, folder_name, f"cluster_token_embeddings_{max_clusters_to_plot}.png"))

def plot_causal_intervention(filepath, num_clusters, distance_threshold, deactivate_strategy="mean", dir=None):
    if not dir:
        dir = f"{VISUALIZATION_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/"
    with open(filepath, "r") as f:
        cluster_id_to_average_MLM_loss = json.load(f)
    cluster_ids = []
    average_MLM_loss_nothing_turned_off = []
    average_MLM_loss_cluster_turned_off = []
    average_MLM_loss_random_turned_off = []
    average_MLM_loss_random_neuron_turned_off = []
    average_MLM_loss_random_layer_dist_neuron_turned_off = []
    for cluster_id in cluster_id_to_average_MLM_loss:
        cluster_ids.append(cluster_id)
        average_MLM_loss_nothing_turned_off.append(cluster_id_to_average_MLM_loss[cluster_id]["nothing_turned_off"])
        average_MLM_loss_cluster_turned_off.append(cluster_id_to_average_MLM_loss[cluster_id]["cluster_turned_off"])
        average_MLM_loss_random_turned_off.append(cluster_id_to_average_MLM_loss[cluster_id]["random_position_dist_neuron_turned_off_mean"])
        average_MLM_loss_random_neuron_turned_off.append(cluster_id_to_average_MLM_loss[cluster_id]["random_neuron_turned_off_mean"])
        average_MLM_loss_random_layer_dist_neuron_turned_off.append(cluster_id_to_average_MLM_loss[cluster_id]["random_layer_dist_neuron_turned_off_mean"])
    print("average_MLM_loss_nothing_turned_off", np.mean(average_MLM_loss_nothing_turned_off))
    print("average_MLM_loss_cluster_turned_off", np.mean(average_MLM_loss_cluster_turned_off))
    print("average_MLM_loss_random_position_turned_off", np.mean(average_MLM_loss_random_turned_off))
    print("average_MLM_loss_random_neuron_turned_off", np.mean(average_MLM_loss_random_neuron_turned_off))
    print("average_MLM_loss_random_layer_dist_neuron_turned_off", np.mean(average_MLM_loss_random_layer_dist_neuron_turned_off))
    plt.figure(figsize=(20, 10))
    plt.scatter(cluster_ids, average_MLM_loss_nothing_turned_off, label="nothing_turned_off", alpha=0.5, marker='^')
    plt.scatter(cluster_ids, average_MLM_loss_cluster_turned_off, label="cluster_turned_off", alpha=0.5, marker='o')
    plt.scatter(cluster_ids, average_MLM_loss_random_turned_off, label="random_position_dist_neuron_turned_off", alpha=0.5, marker='s')
    # plt.scatter(cluster_ids, average_MLM_loss_random_neuron_turned_off, label="random_neuron_turned_off", alpha=0.5)
    # plt.scatter(cluster_ids, average_MLM_loss_random_layer_dist_neuron_turned_off, label="random_layer_dist_neuron_turned_off", alpha=0.5)
    plt.xlabel("Cluster id")
    plt.ylabel("Average MLM loss")
    plt.title("Causal interventions: different MLM loss")
    plt.legend()
    plt.savefig(os.path.join(dir, f"deactivate_{deactivate_strategy}_causal_intervention.png"))


if __name__=="__main__":
    # plot_causal_intervention("c4/cluster_outputs/n_clusters50_distance_threshold_None/deactivate_mean_cluster_id_to_average_MLM_loss.json", 50, None)
    # plot_causal_intervention("c4/cluster_outputs/n_clusters50_distance_threshold_None_tfidf_filter10k/deactivate_mean_cluster_id_to_average_MLM_loss.json", 50, None, dir="c4/visualizations/n_clusters50_distance_threshold_None_tfidf_filter10k")
    # plot_causal_intervention("c4/cluster_outputs_smoothed/n_clusters50_distance_threshold_None/deactivate_mean_cluster_id_to_average_MLM_loss.json", 50, None, dir="c4/visualizations/n_clusters50_distance_threshold_None_smoothed")
    # plot_causal_intervention("c4/cluster_outputs_frequency_only/n_clusters50_distance_threshold_None/deactivate_mean_cluster_id_to_average_MLM_loss.json", 50, None, dir="c4/cluster_outputs_frequency_only/n_clusters50_distance_threshold_None")
    # plot_causal_intervention("c4/cluster_outputs_frequency_only/n_clusters200_distance_threshold_None/deactivate_mean_cluster_id_to_average_MLM_loss.json", 200, None, dir="c4/cluster_outputs_frequency_only/n_clusters200_distance_threshold_None")
    # plot_causal_intervention("c4/cluster_outputs_smoothed/n_clusters50_distance_threshold_None/deactivate_mean_cluster_id_to_average_MLM_loss.json", 50, None, dir="c4/visualizations_smoothed/n_clusters50_distance_threshold_None")
    # plot_causal_intervention("c4/cluster_outputs_smoothed/n_clusters200_distance_threshold_None/deactivate_mean_cluster_id_to_average_MLM_loss.json", 200, None, dir="c4/visualizations_smoothed/n_clusters200_distance_threshold_None")
    # plot_causal_intervention("c4/cluster_outputs_smoothed_refined/n_clusters50_max_dist_0.6/deactivate_mean_cluster_id_to_average_MLM_loss.json", 200, None, dir="c4/visualizations_smoothed_refined/n_clusters50_max_dist_0.6")
    plot_causal_intervention("c4/cluster_outputs_smoothed_refined/n_clusters200_max_dist_0.6/deactivate_mean_cluster_id_to_average_MLM_loss.json", 200, None, dir="c4/visualizations_smoothed_refined/n_clusters200_max_dist_0.6")