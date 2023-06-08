from constants import *
import torch
import json
import os
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_dataset


def load_dataset_from_hf(dev=False):
    if dev:
        yelp = load_dataset("yelp_review_full")
        dataset = yelp["test"]["text"][:10000] # for development purpose, only use the first 10000 examples in yelp["test"]["text"]
    else:
        data_files = {"validation": "en/c4-validation.*.json.gz"}
        dataset = load_dataset("allenai/c4", data_files=data_files, split="validation")
        # dataset = load_dataset("c4", "en", split="validation")
        dataset = dataset["text"]  # Note: using all data
    print("Dataset loaded")
    return dataset

def load_neuron_repr():
    print(f'Loading neuron representations from {NEURON_REPR_DIR}')
    neuron_representations_avg = {}
    for i in tqdm(range(NUM_LAYERS)):
        with open(f'{NEURON_REPR_DIR}/neuron_repr_{i}.json', 'r') as f:
            neuron_representations_avg[i] = torch.tensor(json.load(f)).t()  # shape (vocab_size, num_neurons) -> (num_neurons, vocab_size); num_neurons is hidden_dim

    # concatenate all layers
    all_layer_repr = torch.cat([neuron_representations_avg[i] for i in range(NUM_LAYERS)], dim=0) # (num_layers * num_neurons, vocab_size)
    return all_layer_repr

def load_and_mask_neuron_repr(threshold=1.5):
    all_layer_repr = load_neuron_repr() # (num_layers * num_neurons, vocab_size)
    # set the neuron representation whose absolute value is smaller than the threshold to 0
    all_layer_repr[all_layer_repr.abs() < threshold] = 0
    print(f"Number of tokens that are non zero: {torch.nonzero(all_layer_repr).shape[0]}")
    print(f"Average number of tokens that are non zero per neuron: {torch.nonzero(all_layer_repr).shape[0]/ (NUM_LAYERS * HIDDEN_DIM)}")
    return all_layer_repr

def load_augmented_neuron_repr():
    print(f'Loading augmented neuron representations from {NEURON_REPR_DIR}')
    with open(f'{NEURON_REPR_DIR}/neuron_repr_augmented.json', 'r') as f:
        all_layer_repr = torch.tensor(json.load(f))  # (num_neurons, vocab_size)

    return all_layer_repr


def save_cluster(cluster_labels, num_clusters, distance_threshold):
    dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
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
    dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    with open(os.path.join(dir, 'cluster_id_to_neurons.json'), 'r') as f:
        clusters = json.load(f)
    return clusters

def select_sentences_with_tokens(corpus, top_tokens, tokenizer=None, size=100):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sentences = []
    for sentence in corpus:
        tokenized_sentence = tokenizer.tokenize(sentence)
        for top_token in top_tokens:
            if top_token in tokenized_sentence:
                sentences.append(sentence)
    if len(sentences) < size:
        print(f"Warning: the corpus is too small to sample {size} sentences")
    sentences = random.sample(sentences, min(size, len(sentences)))
    return sentences

def read_top_activating_tokens(filename):
    cluster_to_tokens = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        cluster_id, tokens = line.split(":")
        cluster_id = cluster_id.strip().split(" ")[1]
        tokens = tokens.strip().replace("[", "").replace("]", "").replace("'", "").split(",")
        cluster_to_tokens[cluster_id] = [token.strip() for token in tokens]
    return cluster_to_tokens

def get_layer_indices(neuron_indices):
    # given a list of neuron indices, return a dictionary mapping layer_id to a list of neuron indices in that layer
    layer_indices = defaultdict(list)
    for neuron_index in neuron_indices:
        layer_id = neuron_index // 768
        layer_indices[layer_id].append(neuron_index % 768) 
    return layer_indices

def get_random_layer_indices(num_neurons_to_turn_off):
    # randomly select neurons to turn off given the number of neurons to turn off
    random_neuron_indices = torch.randperm(NUM_LAYERS * HIDDEN_DIM)[:num_neurons_to_turn_off].tolist()
    random_layer_indices = defaultdict(list)
    for neuron_index in random_neuron_indices:
        layer_id = neuron_index // 768
        random_layer_indices[layer_id].append(neuron_index % 768)
    return random_layer_indices

def save_cluster_to_MLM_loss(cluster_id_to_MLM_loss, num_clusters, distance_threshold, deactivate_strategy="zero"):
    dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, f'deactivate_{deactivate_strategy}_cluster_id_to_average_MLM_loss.json'), 'w') as f:
        json.dump(cluster_id_to_MLM_loss, f, indent=4)