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
from utils import *
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

def prepare_inputs_and_labels(batch, top_tokens, tokenizer, config, device, mask_strategy="random", mask_percentage=0.15):
    """
    Args:
        batch: list of sentences
        top_tokens: list of top activating tokens for the cluster
        tokenizer: AutoTokenizer
        config: BertConfig
        device: torch.device
        mask_strategy: mask "random" or "top"-acivating tokens in sentences
        mask_percentage: percentage of tokens to be masked if mask_strategy is "random"
    """
    temp = tokenizer.batch_encode_plus(
                    batch, 
                    add_special_tokens=True,
                    padding=True,
                    truncation=True, 
                    max_length=config.max_position_embeddings, 
                    return_tensors='pt', 
                    return_attention_mask=True)
    input_ids, attention_mask = temp["input_ids"].to(device), temp["attention_mask"].to(device) # shape: (batch_size, seq_len)
    labels = input_ids.clone()
    # set [PAD] tokens to be -100
    labels[labels == tokenizer.pad_token_id] = -100
    if mask_strategy == "random":
        # randomly mask 15% of the input tokens to be [MASK]
        masked_indices = torch.bernoulli(torch.full(input_ids.size(), mask_percentage)).bool().to(device)
        # only mask those tokens that are not [PAD], [CLS], or [SEP]
        masked_indices = masked_indices & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
        # count number of masked indices that is not zero
        # print("Number of masked tokens: ", torch.nonzero(masked_indices).size(0))
        input_ids[masked_indices] = tokenizer.mask_token_id
    elif mask_strategy == "top":
        # TODO: the percentage of tokens being masked is far lower than 15%, maybe should random mask percentage should match this
        # mask top activating tokens
        token_ids = tokenizer.convert_tokens_to_ids(top_tokens)
        masked_indices = torch.zeros(input_ids.size(), dtype=torch.bool).to(device)
        for token_id in token_ids:
            masked_indices = masked_indices | (input_ids == token_id)
        # count number of masked indices that is not zero
        # print("Number of masked tokens: ", torch.nonzero(masked_indices).size(0))
        input_ids[masked_indices] = tokenizer.mask_token_id
    else:
        raise ValueError("mask_strategy must be either 'random' or 'top'")
    # confirm that input_ids is different from labels
    assert not torch.equal(input_ids, labels)
    # MLM loss should ignore indices other than the masked indices
    # set labels to -100 (ignore index) except the masked indices
    labels[~masked_indices] = -100
    return input_ids, attention_mask, labels

def model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, layer_indices, device, random=False):
    """
    Args:
        model: BertForMaskedLM
        config: BertConfig
        input_ids: shape (batch_size, seq_len)
        attention_mask: shape (batch_size, seq_len)
        labels: shape (batch_size, seq_len)
        layer_indices: dict, key is layer id, value is a list of neuron indices in that cluster
        device: torch.device
        random: whether to randomly turn off neurons in the cluster based on the same layer distribution of that cluster
    """
    # define MLM loss function
    loss_fct = CrossEntropyLoss() 

    extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size()).to(device)
    hidden_states= model.bert.embeddings(input_ids=input_ids)
    if random:
        # randomly select neurons to turn off, the number of neurons to turn off in each layer is the same as the number of neurons in the cluster in each layer
        num_neurons_to_turn_off = len(layer_indices[0])
        neuron_indices = torch.randperm(HIDDEN_DIM)[:num_neurons_to_turn_off].tolist()
        hidden_states[:, :, neuron_indices] = 0
    else:
        hidden_states[:, :, layer_indices[0]] = 0 # set the activation of neurons in the embedding layer to 0
    for layer_id in range(1, NUM_LAYERS):
        hidden_states = model.bert.encoder.layer[layer_id - 1](hidden_states, attention_mask=extended_attention_mask)[0]
        # hidden_states has shape (batch_size, sequence_length, hidden_size)
        # for each neuron in the layer, set the activation to 0
        if random:
            # randomly select neurons to turn off, the number of neurons to turn off in each layer is the same as the number of neurons in the cluster in each layer
            num_neurons_to_turn_off = len(layer_indices[layer_id])
            neuron_indices = torch.randperm(HIDDEN_DIM)[:num_neurons_to_turn_off].tolist()
            hidden_states[:, :, neuron_indices] = 0
        else:
            hidden_states[:, :, layer_indices[layer_id]] = 0
    sequence_output = hidden_states # shape: (batch_size, seq_len, hidden_size)
    prediction_scores = model.cls(sequence_output) # shape: (batch_size, sequence_length, vocab_size)
    # compute MLM loss
    masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), labels.view(-1))
    return masked_lm_loss

def evaluate_cluster(num_clusters=3, distance_threshold=None, mask_strategy="top", mask_percentage=0.15, num_repeat=5):
    '''
    Evaluate cluster using causal ablation.

    Args:
        num_clusters: number of clusters, should be None if distance_threshold is not None
        distance_threshold: distance threshold for AgglomerativeClustering, should be None if num_clusters is not None
        mask_strategy: mask "random" or "top"-acivating tokens in sentences
        mask_percentage: percentage of tokens to be masked if mask_strategy is "random"
        num_repeat: number of times to repeat the random baselines
    '''
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

    cluster_id_to_average_MLM_loss = {}
    # for each cluster, for neurons in that cluster, manually set the activation to 0
    for cluster_id in range(num_clusters):
        # get neurons
        neuron_indices = cluster_to_neurons[str(cluster_id)]

        # group by layer id, get a dictionary mapping layer_id to a list of neuron indices in that layer
        layer_indices = get_layer_indices(neuron_indices)
        
        # get randomly selected neuron indices, which is in the range of 0 to NUM_LAYERS * HIDDEN_DIM, the total is equal to the number of neurons in the cluster
        num_neurons_to_turn_off = len(neuron_indices)
        random_layer_indices_lst = []
        for _ in range(num_repeat):
            random_neuron_indices = get_random_layer_indices(num_neurons_to_turn_off)
            random_layer_indices_lst.append(random_layer_indices)
        
        # get tokens & prepare evaluate data
        top_tokens = cluster_to_tokens[str(cluster_id)]
        print("top activating tokens: ", top_tokens)
        evaluation_split = select_sentences_with_tokens(dataset, top_tokens, size=96)
        if len(evaluation_split) == 0:
            print("Warning: no sentence contains the tokens")
            continue

        average_MLM_loss = 0 # with cluster
        average_MLM_loss_random_lst = [0] * num_repeat # random neurons turned off
        average_MLM_loss_random_layer_dist_lst = [0] * num_repeat # random neurons turned off following same layer distribution
        average_MLM_loss_cluster_turned_off = 0 # neurons in cluster turned off
        with torch.no_grad():
            for start_index in tqdm(range(0, len(evaluation_split), BATCH_SIZE)):
                batch = evaluation_split[start_index: start_index+BATCH_SIZE]

                # prepare inputs and labels
                input_ids, attention_mask, labels = prepare_inputs_and_labels(batch, top_tokens, tokenizer, config, device, mask_strategy=mask_strategy, mask_percentage=mask_percentage)

                # turn off neurons in cluster
                masked_lm_loss_cluster_turned_off = model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, layer_indices, device)
                average_MLM_loss_cluster_turned_off += masked_lm_loss_cluster_turned_off.item()

                # turn off random neurons
                for i in range(num_repeat):
                    random_layer_indices = random_layer_indices_lst[i]
                    masked_lm_loss_random = model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, random_layer_indices, device)
                    average_MLM_loss_random_lst[i] += masked_lm_loss_random.item()

                # turn off neurons randomly following same layer distribution
                for i in range(num_repeat):
                    masked_lm_loss_random_layer_dist = model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, layer_indices, device, random=True)
                    average_MLM_loss_random_layer_dist_lst[i] += masked_lm_loss_random_layer_dist.item()
                
                # do not turn off neurons
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                masked_lm_loss = outputs.loss
                average_MLM_loss += masked_lm_loss.item()

        average_MLM_loss /= len(evaluation_split) / BATCH_SIZE
        average_MLM_loss_cluster_turned_off /= len(evaluation_split) / BATCH_SIZE
        average_MLM_loss_random_lst = [item / (len(evaluation_split) / BATCH_SIZE) for item in average_MLM_loss_random_lst]
        average_MLM_loss_random_layer_dist_lst = [item / (len(evaluation_split) / BATCH_SIZE) for item in average_MLM_loss_random_layer_dist_lst]

        cluster_id_to_average_MLM_loss[cluster_id] = {
            "nothing_turned_off": average_MLM_loss,
            "cluster_turned_off": average_MLM_loss_cluster_turned_off,
            "random_neuron_turned_off": average_MLM_loss_random_lst,
            "random_neuron_turned_off_mean": np.mean(average_MLM_loss_random_lst),
            "random_layer_dist_neuron_turned_off": average_MLM_loss_random_layer_dist_lst,
            "random_layer_dist_neuron_turned_off_mean": np.mean(average_MLM_loss_random_layer_dist_lst)
        }
        print("Cluster {}: nothing_turned_off: {}, cluster_turned_off: {}, random_neuron_turned_off: {}, random_layer_dist_neuron_turned_off: {}".format(cluster_id, average_MLM_loss, average_MLM_loss_cluster_turned_off, average_MLM_loss_random_lst, average_MLM_loss_random_layer_dist_lst))
    
    # save cluster_id_to_average_MLM_loss to file
    dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, 'cluster_id_to_average_MLM_loss.json'), 'w') as f:
        json.dump(cluster_id_to_average_MLM_loss, f, indent=4)


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
    evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5)

    # visualize_cluster_token_embeddings(folder_name="n_clusters50_distance_threshold_None")
