# concepts dataset: 1000 common English words https://simple.wikipedia.org/wiki/Wikipedia:List_of_1000_basic_words
# neuron activations (tokens) --> mask out by threshold (both negative and positive)
# concept operations: negation, OR, AND, neighbor
# match neuron with concept: match single if all tokens of the concept are activated; match negation with negative activation; match OR if only the current or the previous part of the concept is activated by neuron; match AND if all parts of the concept are activated; match neighbor if > x% the concept word's topk word2vec neighbors are activated
# compute IoU score for sum over vocab and neurons in the cluster
# build compositional concepts for each cluster with beam search
# at first step, match UNARY(concept); then, at each step, match prev AND/OR UNARY(concept)
# (?) TODO: build concept base that contains different lemmatization of words using e.g. spacy and a large vocabulary

import os
import json
import heapq
import random
import torch
from transformers import AutoTokenizer
from constants import *
import utils

def format_concept_words():
    # TODO: get tokens from concept words using lexical overlaps and overlaps with synonyms

    with open(CONCEPT_WORDS_FILE, 'r') as f:
        lines = f.readlines()
    concept_words = []
    for line in lines:
        # skip lines that start with capital letter
        if len(line.strip()) <= 1:
            continue  
        concept_words.extend(line.strip()[:-1].split(', '))
    # shuffle the order of concept words
    random.shuffle(concept_words)
    print('Number of concept words:', len(concept_words))
    return concept_words

def get_masked_neuron_activations(abs_threshold=0.5, num_clusters=50):
    all_layer_repr = utils.load_neuron_repr()  # (num_layers * num_neurons, vocab_size)
    cluster_id_to_neurons = utils.load_cluster(num_clusters=num_clusters)
    cluster_neuron_reprs = {}
    for cluster_id, neuron_idxs in cluster_id_to_neurons.items():
        cluster_id = int(cluster_id)
        cluster_neuron_reprs[cluster_id] = all_layer_repr[neuron_idxs, :]
        # mask out neuron activations whose absolute value is below threshold
        cluster_neuron_reprs[cluster_id][torch.abs(cluster_neuron_reprs[cluster_id]) < abs_threshold] = 0
        # set positive activations to 1 and negative activations to -1
        cluster_neuron_reprs[cluster_id][cluster_neuron_reprs[cluster_id] > 0] = 1
        cluster_neuron_reprs[cluster_id][cluster_neuron_reprs[cluster_id] < 0] = -1

    return cluster_neuron_reprs

def compute_iou_score(cluster_rep, concept, tokenizer):
    '''compute IoU score for the cluster and the concept
    cluster_rep: shape (vocab_size, num_neurons)
    '''
    concept_rep = torch.zeros_like(cluster_rep)
    # TODO: expand the concept with its synonyms
    for word_or_op in concept:
        if word_or_op in ['AND']:
            continue
        current_concept_word = word_or_op
        not_flag = False
        if current_concept_word.startswith('NOT '):
            current_concept_word = current_concept_word[4:]
            not_flag = True
        concept_tokens = tokenizer(current_concept_word, return_tensors='pt')['input_ids'][0]
        # TODO: cut of the first and last token (CLS and SEP)
        concept_rep[:, concept_tokens] = -1 if not_flag else 1

    intersection = torch.sum(cluster_rep * concept_rep)
    union = torch.sum(torch.max(cluster_rep.abs(), concept_rep.abs()))
    iou_score = intersection / union
    return iou_score.item()


def build_concept_for_cluster(cluster_rep, concept_words, tokenizer, beam_size=1, max_len=5, num_concepts_to_return=1):
    '''build compositional concept for a cluster with beam search
    cluster_rep: shape (num_neurons, vocab_size)
    concept_words: list of concept words
    tokenizer: tokenizer
    beam_size: beam size
    return a list of strings representing the compositional concept (e.g. [NOT a, AND, b])
    '''
    concept_beam = [(0, [])]
    negated_concept_words = ['NOT ' + word for word in concept_words]
    for _ in range(max_len):
        new_beam = []
        for score, concept in concept_beam:
            for concept_word in concept_words + negated_concept_words:
                if len(concept) > 0:
                    for op in ['AND']:  # ignore OR for now
                        new_concept = concept + [op, concept_word]
                        new_score = compute_iou_score(cluster_rep, new_concept, tokenizer)
                        new_beam.append((new_score, new_concept))
                else:
                    new_concept = concept + [concept_word]
                    new_score = compute_iou_score(cluster_rep, new_concept, tokenizer)
                    heapq.heappush(new_beam, (new_score, new_concept))
        # keep only the top beam_size concepts
        concept_beam = heapq.nlargest(beam_size, new_beam)
        print(concept_beam)

    # return the top 5 concept with the highest score
    top_concepts = [(score, concept) for score, concept in concept_beam[:num_concepts_to_return]]
    return top_concepts


def main(num_clusters=50):
    concept_words = format_concept_words()
    print("Getting neuron activations...")
    cluster_neuron_reprs = get_masked_neuron_activations(abs_threshold=1, num_clusters=num_clusters)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for i in range(num_clusters):
        # build compositional concepts for each cluster with beam search
        print(f"Building compositional concepts for cluster {i}...")
        cluster_rep = cluster_neuron_reprs[i]
        concept = build_concept_for_cluster(cluster_rep, concept_words, tokenizer)
        print(concept)
        # save the compositional concept
        with open(f'{CONCEPTS_DIR}/concept.json', 'a') as f:
            json.dump({i: concept}, f)
            f.write('\n')


if __name__ == "__main__":
    main()
