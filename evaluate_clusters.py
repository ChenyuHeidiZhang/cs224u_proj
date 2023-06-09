from tqdm import tqdm
import numpy as np
import os
import json
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, BertModel, BertConfig, BertForPreTraining, BertForMaskedLM

from constants import *
import utils

def prepare_inputs_and_labels(batch, top_tokens, tokenizer, config, device, mask_strategy="random", mask_percentage=0.05):
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
    seq_len = input_ids.size(1)
    labels = input_ids.clone()
    # set [PAD] tokens to be -100
    labels[labels == tokenizer.pad_token_id] = -100
    if mask_strategy == "random":
        # random mask percentage should match this mask top activating tokens
        token_ids = tokenizer.convert_tokens_to_ids(top_tokens)
        masked_indices = torch.zeros(input_ids.size(), dtype=torch.bool).to(device)
        for token_id in token_ids:
            masked_indices = masked_indices | (input_ids == token_id)
        # count number of masked indices that is not zero
        masked_count = torch.nonzero(masked_indices).size(0)
        mask_percentage = masked_count / (input_ids.size(0) * input_ids.size(1))
        print("mask_percentage: ", mask_percentage)
        # randomly mask 15% of the input tokens to be [MASK]
        masked_indices = torch.bernoulli(torch.full(input_ids.size(), mask_percentage * 2)).bool().to(device)
        # only mask those tokens that are not [PAD], [CLS], or [SEP]
        masked_indices = masked_indices & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
        # count number of masked indices that is not zero
        # print("Number of masked tokens: ", torch.nonzero(masked_indices).size(0))
        input_ids[masked_indices] = tokenizer.mask_token_id
        print("non zero masked indices: ", torch.nonzero(masked_indices).size(0))
    elif mask_strategy == "top":
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
    # assert not torch.equal(input_ids, labels)
    # MLM loss should ignore indices other than the masked indices
    # set labels to -100 (ignore index) except the masked indices
    labels[~masked_indices] = -100
    return input_ids, attention_mask, labels

def model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, layer_indices, device, random=False, random_strategy="layer", deactivate_strategy="zero"):
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
        random_strategy: "layer" or "position"
        deactivate_strategy: "zero" or "mean" of hidden state for an example when deactivating neurons in a cluster
    """
    # define MLM loss function
    loss_fct = CrossEntropyLoss() 

    extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size()).to(device)
    hidden_states= model.bert.embeddings(input_ids=input_ids)

    # get total number of neurons in the cluster
    num_neurons = sum([len(layer_indices[layer_id]) for layer_id in layer_indices])
    # get average number of neurons per layer
    average_num_neurons_per_layer = num_neurons // NUM_LAYERS
    # random select average_num_neurons_per_layer neurons from the range of 0 to HIDDEN_DIM
    positions = torch.randperm(HIDDEN_DIM)[:average_num_neurons_per_layer].tolist()
    
    if random:
        if random_strategy=="layer":
            # randomly select neurons to turn off, the number of neurons to turn off in each layer is the same as the number of neurons in the cluster in each layer
            num_neurons_to_turn_off = len(layer_indices[0])
            neuron_indices = torch.randperm(HIDDEN_DIM)[:num_neurons_to_turn_off].tolist()
            hidden_states[:, :, neuron_indices] = 0 if deactivate_strategy == "zero" else torch.mean(hidden_states, dim=2, keepdim=True)[:, :, 0][:, :, None]
        elif random_strategy=="position":
            hidden_states[:, :, positions] = 0 if deactivate_strategy == "zero" else torch.mean(hidden_states, dim=2, keepdim=True)[:, :, 0][:, :, None]
    else:
        # set the activation of neurons in the embedding layer to 0
        hidden_states[:, :, layer_indices[0]] = 0 if deactivate_strategy == "zero" else torch.mean(hidden_states, dim=2, keepdim=True)[:, :, 0][:, :, None]
    for layer_id in range(1, NUM_LAYERS):
        hidden_states = model.bert.encoder.layer[layer_id - 1](hidden_states, attention_mask=extended_attention_mask)[0]
        # hidden_states has shape (batch_size, sequence_length, hidden_size)
        # for each neuron in the layer, set the activation to 0
        if random:
            if random_strategy=="layer":
                # randomly select neurons to turn off, the number of neurons to turn off in each layer is the same as the number of neurons in the cluster in each layer
                num_neurons_to_turn_off = len(layer_indices[layer_id])
                neuron_indices = torch.randperm(HIDDEN_DIM)[:num_neurons_to_turn_off].tolist()
                hidden_states[:, :, neuron_indices] = 0 if deactivate_strategy == "zero" else torch.mean(hidden_states, dim=2, keepdim=True)[:, :, 0][:, :, None]
            elif random_strategy=="position":
                hidden_states[:, :, positions] = 0 if deactivate_strategy == "zero" else torch.mean(hidden_states, dim=2, keepdim=True)[:, :, 0][:, :, None]
        else:
            # print("set mean to: ", torch.mean(hidden_states, dim=2, keepdim=True)[:, :, 0][:, :, None].shape)
            hidden_states[:, :, layer_indices[layer_id]] = 0 if deactivate_strategy == "zero" else torch.mean(hidden_states, dim=2, keepdim=True)[:, :, 0][:, :, None]
    sequence_output = hidden_states # shape: (batch_size, seq_len, hidden_size)
    prediction_scores = model.cls(sequence_output) # shape: (batch_size, sequence_length, vocab_size)
    # compute MLM loss
    masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), labels.view(-1))
    return masked_lm_loss

def evaluate_cluster(num_clusters=3, distance_threshold=None, mask_strategy="top", mask_percentage=0.15, num_repeat=5, evaluation_size=96, deactivate_strategy="zero", dir=None):
    '''
    Evaluate cluster using causal ablation.

    Args:
        num_clusters: number of clusters, should be None if distance_threshold is not None
        distance_threshold: distance threshold for AgglomerativeClustering, should be None if num_clusters is not None
        mask_strategy: mask "random" or "top"-acivating tokens in sentences
        mask_percentage: percentage of tokens to be masked if mask_strategy is "random"
        num_repeat: number of times to repeat the random baselines
        evaluation_size: number of sentences to evaluate on per cluster
        deactivate_strategy: "zero" or "mean" of hidden state for an example when deactivating neurons in a cluster
    '''
    # load neurons in each cluster
    if dir:
        cluster_to_neurons = utils.load_cluster_from_file(os.path.join(dir, 'cluster_id_to_neurons.json'))
        # load top activating tokens for each cluster
        # cluster_to_tokens = utils.read_top_activating_tokens(os.path.join(dir, 'top_10_tokens.txt'))
        cluster_to_tokens = utils.read_top_activating_tokens(os.path.join(dir, 'top_30_tokens.txt'))
    else:
        cluster_to_neurons = utils.load_cluster(num_clusters=num_clusters, distance_threshold=distance_threshold)
        cluster_to_tokens = utils.read_top_activating_tokens(f"{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/top_10_tokens.txt")
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
    dataset = utils.load_dataset_from_hf(dev=(DATASET=='yelp'))

    # check if os.path.join(dir, 'cluster_id_to_evaluation_split.json') exists if exists ,load it
    if os.path.exists(os.path.join(dir, 'cluster_id_to_evaluation_split.json')):
        print("Start loading evaluation split for clusters")
        with open(os.path.join(dir, 'cluster_id_to_evaluation_split.json'), 'r') as f:
            cluster_to_evaluation_split = json.load(f)
        print("Done loading evaluation split for clusters")
    # if not, select evaluation split for clusters
    else:
        print("Start selecting evaluation split for clusters")
        cluster_to_evaluation_split = utils.select_sentences_for_all_clusters(dataset, cluster_to_tokens, tokenizer=tokenizer, size=evaluation_size)
        print("Done selecting evaluation split for clusters")

        # write evaluation split to file
        if not dir:
            dir = f'{CLUSTER_OUTPUT_DIR}/n_clusters{num_clusters}_distance_threshold_{distance_threshold}/'
        with open(os.path.join(dir, 'cluster_id_to_evaluation_split.json'), 'w') as f:
            json.dump(cluster_to_evaluation_split, f, indent=4)

    cluster_id_to_average_MLM_loss = {}
    # for each cluster, for neurons in that cluster, manually set the activation to 0
    for cluster_id in range(num_clusters):
        # get neurons
        neuron_indices = cluster_to_neurons[str(cluster_id)]

        # group by layer id, get a dictionary mapping layer_id to a list of neuron indices in that layer
        layer_indices = utils.get_layer_indices(neuron_indices)
        
        # get randomly selected neuron indices, which is in the range of 0 to NUM_LAYERS * HIDDEN_DIM, the total is equal to the number of neurons in the cluster
        num_neurons_to_turn_off = len(neuron_indices)
        random_layer_indices_lst = []
        for _ in range(num_repeat):
            random_layer_indices = utils.get_random_layer_indices(num_neurons_to_turn_off)
            random_layer_indices_lst.append(random_layer_indices)
        
        # get tokens & prepare evaluate data
        top_tokens = cluster_to_tokens[str(cluster_id)]
        print("top activating tokens: ", top_tokens)
        # evaluation_split = utils.select_sentences_with_tokens(dataset, top_tokens, size=evaluation_size)
        evaluation_split = cluster_to_evaluation_split[str(cluster_id)]
        print("done selecting sentences for evaluation")
        if len(evaluation_split) == 0:
            print("Warning: no sentence contains the tokens")
            continue

        average_MLM_loss = 0 # with cluster, predict top activating tokens
        average_MLM_loss_random_tokens = 0 # with_cluster, predict random tokens
        average_MLM_loss_random_position_dist_lst = [0] * num_repeat # random neurons turned off following same position distribution
        average_MLM_loss_random_lst = [0] * num_repeat # random neurons turned off
        average_MLM_loss_random_layer_dist_lst = [0] * num_repeat # random neurons turned off following same layer distribution
        average_MLM_loss_cluster_turned_off = 0 # neurons in cluster turned off
        average_MLM_loss_cluster_turned_off_random_tokens = 0 # neurons in cluster turned off, random tokens turned off
        with torch.no_grad():
            for start_index in tqdm(range(0, len(evaluation_split), BATCH_SIZE)):
                batch = evaluation_split[start_index: start_index+BATCH_SIZE]

                # prepare inputs and labels that masks top activating tokens
                input_ids, attention_mask, labels = prepare_inputs_and_labels(batch, top_tokens, tokenizer, config, device, mask_strategy=mask_strategy, mask_percentage=mask_percentage)

                # prepare inputs and labels that masks random tokens
                random_mask_input_ids, random_mask_attention_mask, random_mask_labels = prepare_inputs_and_labels(batch, top_tokens, tokenizer, config, device, mask_strategy="random", mask_percentage=0.05)

                # do not turn off neurons: predict top activating tokens
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                masked_lm_loss = outputs.loss
                average_MLM_loss += masked_lm_loss.item()

                # do not turn off neurons: predict random tokens
                outputs = model(input_ids=random_mask_input_ids, attention_mask=random_mask_attention_mask, labels=random_mask_labels)
                masked_lm_loss_random_tokens = outputs.loss
                average_MLM_loss_random_tokens += masked_lm_loss_random_tokens.item()


                # turn off neurons in cluster: predict top activating tokens
                masked_lm_loss_cluster_turned_off = model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, layer_indices, device, deactivate_strategy=deactivate_strategy)
                average_MLM_loss_cluster_turned_off += masked_lm_loss_cluster_turned_off.item()

                # # turn off neurons in cluster: predict random tokens 
                masked_lm_loss_cluster_turned_off_random_tokens = model_forward_cluster_turned_off(model, config, random_mask_input_ids, random_mask_attention_mask, random_mask_labels, layer_indices, device, deactivate_strategy=deactivate_strategy)
                average_MLM_loss_cluster_turned_off_random_tokens += masked_lm_loss_cluster_turned_off_random_tokens.item()


                # turn off neurons randomly following the same position distribution
                for i in range(num_repeat):
                    masked_lm_loss_random_position_dist = model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, layer_indices, device, random=True, random_strategy="position", deactivate_strategy=deactivate_strategy)
                    average_MLM_loss_random_position_dist_lst[i] += masked_lm_loss_random_position_dist.item()

                # turn off random neurons
                for i in range(num_repeat):
                    random_layer_indices = random_layer_indices_lst[i]
                    masked_lm_loss_random = model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, random_layer_indices, device, deactivate_strategy=deactivate_strategy)
                    average_MLM_loss_random_lst[i] += masked_lm_loss_random.item()

                # turn off neurons randomly following same layer distribution
                for i in range(num_repeat):
                    masked_lm_loss_random_layer_dist = model_forward_cluster_turned_off(model, config, input_ids, attention_mask, labels, layer_indices, device, random=True, random_strategy="layer", deactivate_strategy=deactivate_strategy)
                    average_MLM_loss_random_layer_dist_lst[i] += masked_lm_loss_random_layer_dist.item()

                
        average_MLM_loss /= len(evaluation_split) / BATCH_SIZE
        average_MLM_loss_cluster_turned_off /= len(evaluation_split) / BATCH_SIZE
        average_MLM_loss_cluster_turned_off_random_tokens /= len(evaluation_split) / BATCH_SIZE
        average_MLM_loss_random_position_dist_lst = [item / (len(evaluation_split) / BATCH_SIZE) for item in average_MLM_loss_random_position_dist_lst]
        average_MLM_loss_random_lst = [item / (len(evaluation_split) / BATCH_SIZE) for item in average_MLM_loss_random_lst]
        average_MLM_loss_random_layer_dist_lst = [item / (len(evaluation_split) / BATCH_SIZE) for item in average_MLM_loss_random_layer_dist_lst]

        cluster_id_to_average_MLM_loss[cluster_id] = {
            "nothing_turned_off": average_MLM_loss,
            "nothing_turned_off_random_tokens": average_MLM_loss_random_tokens,
            "cluster_turned_off": average_MLM_loss_cluster_turned_off,
            "cluster_turned_off_random_tokens": average_MLM_loss_cluster_turned_off_random_tokens,
            "random_position_dist_neuron_turned_off": average_MLM_loss_random_position_dist_lst,
            "random_position_dist_neuron_turned_off_mean": np.mean(average_MLM_loss_random_position_dist_lst),
            "random_neuron_turned_off": average_MLM_loss_random_lst,
            "random_neuron_turned_off_mean": np.mean(average_MLM_loss_random_lst),
            "random_layer_dist_neuron_turned_off": average_MLM_loss_random_layer_dist_lst,
            "random_layer_dist_neuron_turned_off_mean": np.mean(average_MLM_loss_random_layer_dist_lst)
        }
        print(f"Cluster {cluster_id}: nothing_turned_off: {average_MLM_loss}, nothing_turned_off_random_tokens: {average_MLM_loss_random_tokens}, cluster_turned_off: {average_MLM_loss_cluster_turned_off}, cluster_turned_off_random_tokens: {average_MLM_loss_cluster_turned_off_random_tokens}, random_position_dist_neuron_turned_off: {average_MLM_loss_random_position_dist_lst}, random_neuron_turned_off: {average_MLM_loss_random_lst}, random_layer_dist_neuron_turned_off: {average_MLM_loss_random_layer_dist_lst}")

    # save cluster_id_to_average_MLM_loss to file
    utils.save_cluster_to_MLM_loss(cluster_id_to_average_MLM_loss, num_clusters, distance_threshold, deactivate_strategy=deactivate_strategy, dir=dir)


if __name__ == "__main__":
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=32, deactivate_strategy="mean", dir = 'c4/cluster_outputs/n_clusters50_distance_threshold_None_tfidf_filter10k')
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=32, deactivate_strategy="mean", dir = 'c4/cluster_outputs_smoothed/n_clusters50_distance_threshold_None')
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=32, deactivate_strategy="mean", dir = 'c4/cluster_outputs/n_clusters50_distance_threshold_None_tfidf')
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=32, deactivate_strategy="mean", dir = 'c4/cluster_outputs/n_clusters50_distance_threshold_None')
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=32, deactivate_strategy="mean", dir = 'c4/cluster_outputs_smoothed/n_clusters50_distance_threshold_None_tfidf')
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=96, deactivate_strategy="mean", dir = 'c4/cluster_outputs_frequency_only/n_clusters50_distance_threshold_None')
    # evaluate_cluster(num_clusters=200, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=96, deactivate_strategy="mean", dir = 'c4/cluster_outputs_frequency_only/n_clusters200_distance_threshold_None')
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=96, deactivate_strategy="mean", dir = 'c4/cluster_outputs_smoothed/n_clusters50_distance_threshold_None')
    # evaluate_cluster(num_clusters=200, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=96, deactivate_strategy="mean", dir = 'c4/cluster_outputs_smoothed/n_clusters200_distance_threshold_None')
    # evaluate_cluster(num_clusters=50, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=96, deactivate_strategy="mean", dir = 'c4/cluster_outputs_smoothed_refined/n_clusters50_max_dist_0.6')
    evaluate_cluster(num_clusters=200, distance_threshold=None, mask_strategy="top", num_repeat=5, evaluation_size=96, deactivate_strategy="mean", dir = 'c4/cluster_outputs_smoothed_refined/n_clusters200_max_dist_0.6')

    # with open('c4/cluster_outputs_frequency_only/n_clusters50_distance_threshold_None/cluster_id_to_evaluation_split.json', 'r') as f:
    #     cluster_to_evaluation_split = json.load(f)
    # # print out len of evaluation split for each cluster
    # for cluster_id in range(50):
    #     print(cluster_id, len(cluster_to_evaluation_split[str(cluster_id)]))
