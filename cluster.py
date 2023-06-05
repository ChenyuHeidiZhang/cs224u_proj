from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torch.nn import CrossEntropyLoss
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, BertModel, BertConfig, BertForPreTraining, BertForMaskedLM

from neuron import load_dataset_from_hf
from constants import *
from utils import load_neuron_repr, save_cluster, load_cluster, read_top_activating_tokens, select_sentences_with_tokens
from visualization import *


def explore_cluster_distance_thresholds(dissimilarity, thresholds):
    for threshold in thresholds:
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=threshold)
        cluster_labels = clustering.fit_predict(dissimilarity) # cluster label for each neuron
        cluster_size = max(cluster_labels) + 1
        print(f"Threshold: {threshold}, Number of clusters: {cluster_size}")

def get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens):
    dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
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
    save_cluster(cluster_labels, num_clusters, distance_threshold)
    # print(cluster_labels)

    # Find top-k tokens that are activated by neurons in the same cluster and write to a file
    cluster_id_to_top_token_indices = get_cluster_top_tokens(all_layer_repr, tokenizer, cluster_labels, num_clusters, distance_threshold, num_top_tokens)

    # plot the positions (layer and index) of neurons for each cluster label
    plot_cluster_neurons(cluster_labels, num_clusters, distance_threshold)

    # TODO: this should really be done across clusters, not within clusters, to show that the clusters are well-separated
    # plot the top tokens for each cluster with their representations
    # plot_cluster_top_tokens_neuron(cluster_id_to_top_token_indices, all_layer_repr, cluster_labels, num_clusters, distance_threshold, num_top_tokens)


def evaluate_cluster(num_clusters=3, distance_threshold=None, mask_percentage=0.15):
    '''Evaluate cluster using causal ablation.'''
    # load neurons in each cluster
    cluster_to_neurons = load_cluster(num_clusters=num_clusters, distance_threshold=distance_threshold)
    # load top activating tokens for each cluster
    cluster_to_tokens = read_top_activating_tokens(f"{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/top_10_tokens.txt")
    print("Cluster loaded")

    # load model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained('tokenizer_info')
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config).to(device)
    print("Model loaded")

    # load data
    dataset = load_dataset_from_hf(dev=(DATASET=='yelp'))
    print("Dataset loaded")

    # define loss function
    loss_fct = CrossEntropyLoss() # MLM

    # for each cluster, for neurons in that cluster, manually set the activation to 0
    for cluster_id in range(num_clusters):
        # get neurons
        neuron_indices = cluster_to_neurons[str(cluster_id)]
        # group by layer id
        layer_indices = defaultdict(list)
        for neuron_index in neuron_indices:
            layer_id = neuron_index // 768
            layer_indices[layer_id].append(neuron_index % 768)
        
        # get tokens & prepare evaluate data
        tokens = cluster_to_tokens[str(cluster_id)]
        print("tokens: ", tokens)
        evaluation_split = select_sentences_with_tokens(dataset, tokens, size=1000)

        average_MLM_loss = 0
        with torch.no_grad():
            for start_index in tqdm(range(0, len(evaluation_split), BATCH_SIZE)):
                batch = dataset[start_index: start_index+BATCH_SIZE]

                temp = tokenizer.batch_encode_plus(
                    batch, add_special_tokens=True, padding=True, truncation=True, max_length=config.max_position_embeddings, return_tensors='pt', return_attention_mask=True)
                input_ids, attention_mask = temp["input_ids"].to(device), temp["attention_mask"].to(device) # shape: (batch_size, seq_len)
                labels = input_ids.clone()
                # set [PAD] tokens to be -100
                labels[labels == tokenizer.pad_token_id] = -100
                # randomly mask 15% of the input tokens to be [MASK]
                masked_indices = torch.bernoulli(torch.full(input_ids.size(), mask_percentage)).bool().to(device)
                # only mask those tokens that are not [PAD], [CLS], or [SEP]
                masked_indices = masked_indices & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
                input_ids[masked_indices] = tokenizer.mask_token_id
                # confirm that input_ids is different from labels
                assert not torch.equal(input_ids, labels)

                extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size()).to(device)
                
                hidden_states= model.bert.embeddings(input_ids=input_ids)
                hidden_states[:, :, layer_indices[0]] = 0 # set the activation of neurons in the embedding layer to 0
                for layer_id in range(1, NUM_LAYERS):
                    hidden_states = model.bert.encoder.layer[layer_id - 1](hidden_states, attention_mask=extended_attention_mask)[0]
                    # hidden_states has shape (batch_size, sequence_length, hidden_size)
                    # for each neuron in the layer, set the activation to 0
                    hidden_states[:, :, layer_indices[layer_id]] = 0
                sequence_output = hidden_states # shape: (batch_size, seq_len, hidden_size)
                prediction_scores = model.cls(sequence_output) # shape: (batch_size, sequence_length, vocab_size)
                # compute MLM loss
                masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), labels.view(-1))
                average_MLM_loss += masked_lm_loss.item()
        average_MLM_loss /= len(evaluation_split) / BATCH_SIZE
        print(f"Cluster {cluster_id} average MLM loss: {average_MLM_loss}")
        # return

    # TODO: compute the accuracy of the model on the dataset with and without the cluster


def run():
    all_layer_repr = load_neuron_repr()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # one of num_cluster and distance_threshold must be None
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=20, distance_threshold=None, num_top_tokens=10)
    compute_clusters(all_layer_repr, tokenizer, num_clusters=50, distance_threshold=None, num_top_tokens=10)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=500, distance_threshold=None, num_top_tokens=10)
    # compute_clusters(all_layer_repr, tokenizer, num_clusters=None, distance_threshold=0.999, num_top_tokens=10)


if __name__ == '__main__':
    # run()
    evaluate_cluster(num_clusters=50, distance_threshold=None)

    # visualize_cluster_token_embeddings(folder_name="n_clusters50_distance_threshold_None")
