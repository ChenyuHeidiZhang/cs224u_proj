import os
import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, BertModel, BertConfig
import fasttext.util

from utils import load_dataset_from_hf, load_and_mask_neuron_repr
from constants import *

def get_neuron_representations(model, tokenizer, config, dataset, device):
    neuron_representations_sum = {}
    # the size of the representation for each neuron is VOCAB_SIZE
    for i in range(config.num_hidden_layers+1):
        neuron_representations_sum[i] = torch.zeros(config.vocab_size, config.hidden_size).to(device)
    tokens_count = torch.zeros(config.vocab_size).to(device)

    with torch.no_grad():
        for start_index in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch = dataset[start_index: start_index+BATCH_SIZE]
            temp = tokenizer.batch_encode_plus(
                batch, add_special_tokens=True, padding=True, truncation=True, max_length=config.max_position_embeddings, return_tensors='pt', return_attention_mask=True)
            input_ids, attention_mask = temp["input_ids"].to(device), temp["attention_mask"].to(device)
            # Note: [PAD] is treated as a normal token here, so we may want to drop the
            # scalar (index: 0) corresponding to [PAD] in the vector representation of any neuron.
            # Alternatively, we may want to skip [PAD] when doing the following computation because 
            # around half of the tokens are [PAD], but I didn't figure out how to do it
            # without a loop
            input_ids_flatten = input_ids.flatten()
            # print(input_ids_flatten.shape)
            tokens_count.index_add_(
                dim=0, index=input_ids_flatten, source=torch.ones_like(input_ids_flatten, dtype=tokens_count.dtype, device=device))
            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]
            for idx, hidden_state in enumerate(hidden_states):
                # hidden_states has shape (BATCH_SIZE, MAX_LENGTH, HIDDEN_SIZE)
                neuron_representations_sum[idx].index_add_(
                    dim=0, index=input_ids_flatten, source=hidden_state.flatten(start_dim=0, end_dim=1))
                # print(hidden_state.flatten(start_dim=0, end_dim=1).shape)

    neuron_representations_avg = {}
    # the size of the representation for each neuron is VOCAB_SIZE
    for i in range(config.num_hidden_layers+1):
        neuron_representations_avg[i] = neuron_representations_sum[i] / torch.clamp(tokens_count.unsqueeze(1), min=1) # avoid division by zero error

    # save neuron representations to file
    if not os.path.exists(NEURON_REPR_DIR):
        os.makedirs(NEURON_REPR_DIR)
    for i in range(config.num_hidden_layers+1):
        neuron_representations_avg[i] = neuron_representations_avg[i].cpu().tolist()
        with open(f'{NEURON_REPR_DIR}/neuron_repr_{i}.json', 'w') as f:
            json.dump(neuron_representations_avg[i], f)

    print('non-zero tokens count: ', torch.nonzero(tokens_count).shape)

    return neuron_representations_avg


def augment_neuron_repr_with_token_similarity(tokenizer, topk_neigh=5, score_discount=0.5):
    fasttext.util.download_model('en', if_exists='ignore')  # English
    print("Loading FastText model...")
    ft = fasttext.load_model('cc.en.300.bin')

    # load all vocab
    vocab = tokenizer.get_vocab()  # token to id dict
    # load neuron representations
    all_layer_repr = load_and_mask_neuron_repr()  # (all_num_neurons, vocab_size)
    # for each token in the vocab, find the topk most similar tokens by fasttext and their similarity scores
    all_layer_repr_aug = all_layer_repr.clone()
    for token_id in tqdm(range(all_layer_repr.size(1))):
        # for each token in the vocab, find the topk most similar tokens by fasttext and their similarity scores
        token = tokenizer.convert_ids_to_tokens(token_id)
        neighbors = ft.get_nearest_neighbors(token)
        selected_neighbor_indices = []
        selected_neighbor_scores = []
        for (score, neigh) in neighbors:
            if neigh in vocab:
                selected_neighbor_indices.append(vocab[neigh])
                selected_neighbor_scores.append(score * score_discount)
            if len(selected_neighbor_indices) == topk_neigh:
                break
        # print(token, selected_neighbor_indices)
        # add the similarity_score * activation of each of the topk neighbor tokens to the current neuron representation
        all_layer_repr_aug[:, token_id] += torch.sum(all_layer_repr[:, selected_neighbor_indices] * torch.tensor(selected_neighbor_scores).unsqueeze(0), dim=1)

    # save the augmented neuron representations to file
    save_path = os.path.join(NEURON_REPR_DIR, 'neuron_repr_augmented.json')
    if not os.path.exists(NEURON_REPR_DIR):
        os.makedirs(NEURON_REPR_DIR)
    all_layer_repr_aug = all_layer_repr_aug.tolist()
    with open(save_path, 'w') as f:
        json.dump(all_layer_repr_aug, f)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained('tokenizer_info')
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = BertModel.from_pretrained("bert-base-uncased", config=config).to(device)
    print("Model loaded")

    # VOCAB_SIZE = config.vocab_size # including added tokens
    # MAX_LENGTH = config.max_position_embeddings
    # NUM_HIDDEN_LAYERS = config.num_hidden_layers
    # HIDDEN_SIZE = config.hidden_size
    print('VOCAB_SIZE:', config.vocab_size, 'MAX_LENGTH:', config.max_position_embeddings, 'NUM_HIDDEN_LAYERS:', config.num_hidden_layers, 'HIDDEN_SIZE:', config.hidden_size)

    dataset = load_dataset_from_hf(dev=(DATASET=='yelp'))
    print("Dataset loaded")

    print("Start computing neuron representations")
    get_neuron_representations(model, tokenizer, config, dataset, device)


if __name__ == "__main__":
    # main()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    augment_neuron_repr_with_token_similarity(tokenizer, topk_neigh=5, score_discount=0.5)