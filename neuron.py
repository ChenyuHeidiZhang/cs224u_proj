import os
import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, BertModel, BertConfig
import fasttext.util

from utils import load_dataset_from_hf, load_and_mask_neuron_repr, load_neuron_repr
from constants import *

def get_neuron_representations(model, tokenizer, config, dataset, device, token_count_only=False):
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
            if not token_count_only:
                outputs = model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs[2]
                for idx, hidden_state in enumerate(hidden_states):
                    # hidden_states has shape (BATCH_SIZE, MAX_LENGTH, HIDDEN_SIZE)
                    neuron_representations_sum[idx].index_add_(
                        dim=0, index=input_ids_flatten, source=hidden_state.flatten(start_dim=0, end_dim=1))
                    # print(hidden_state.flatten(start_dim=0, end_dim=1).shape)

    # save token count to file
    print('non-zero tokens count: ', torch.nonzero(tokens_count).shape)
    print('mean tokens count: ', torch.mean(tokens_count))
    print('median tokens count: ', torch.median(tokens_count))
    print('max tokens count: ', torch.max(tokens_count))
    print('std tokens count: ', torch.std(tokens_count))

    if not os.path.exists(NEURON_REPR_DIR):
        os.makedirs(NEURON_REPR_DIR)
    tokens_count = tokens_count.cpu().tolist()
    with open(f'{NEURON_REPR_DIR}/tokens_count.json', 'w') as f:
        json.dump(tokens_count, f)
    if token_count_only:
        return None

    neuron_representations_avg = {}
    # the size of the representation for each neuron is VOCAB_SIZE
    for i in range(config.num_hidden_layers+1):
        neuron_representations_avg[i] = neuron_representations_sum[i] / torch.clamp(tokens_count.unsqueeze(1), min=1) # avoid division by zero error

    # save neuron representations to file
    for i in range(config.num_hidden_layers+1):
        neuron_representations_avg[i] = neuron_representations_avg[i].cpu().tolist()
        with open(f'{NEURON_REPR_DIR}/neuron_repr_{i}.json', 'w') as f:
            json.dump(neuron_representations_avg[i], f)

    return neuron_representations_avg


def filter_neuron_repr_with_token_count(min_count=50):
    with open(f'{NEURON_REPR_DIR}/tokens_count.json', 'r') as f:
        tokens_count = json.load(f)
    token_ids_to_keep = [idx for idx, count in enumerate(tokens_count) if count >= min_count]
    # remove [PAD], [CLS], [SEP]
    token_ids_to_keep = [idx for idx in token_ids_to_keep if idx not in [0, 101, 102]]
    print('token_ids_to_keep: ', len(token_ids_to_keep))

    # save token_ids_to_keep to file
    with open(f'{NEURON_REPR_DIR}/token_ids_to_keep.json', 'w') as f:
        json.dump(token_ids_to_keep, f)

    for i in tqdm(range(NUM_LAYERS)):
        with open(f'{NEURON_REPR_DIR}/neuron_repr_{i}.json', 'r') as f:
            neuron_repr = json.load(f)
        neuron_repr = [neuron_repr[idx] for idx in token_ids_to_keep]
        with open(f'{NEURON_REPR_DIR}/neuron_repr_{i}_filtered.json', 'w') as f:
            json.dump(neuron_repr, f)

    # yelp vocab size for min_count=10: 6693
    # c4 vocab size for min_count=400: 15298


def augment_neuron_repr_with_token_similarity(tokenizer, topk_neigh=5, score_discount=0.5):
    fasttext.util.download_model('en', if_exists='ignore')  # English
    print("Loading FastText model...")
    ft = fasttext.load_model('cc.en.300.bin')

    # load all vocab
    vocab = tokenizer.get_vocab()  # token to id dict
    # load neuron representations
    all_layer_repr = load_neuron_repr()  # (all_num_neurons, vocab_size)
    with open(f'{NEURON_REPR_DIR}/token_ids_to_keep.json', 'r') as f:
        token_ids_kept = json.load(f)

    all_layer_repr_aug = all_layer_repr.clone()
    for i, token_id in enumerate(tqdm(token_ids_kept)):
        # for each token in the vocab, find the topk most similar tokens by fasttext and their similarity scores
        token = tokenizer.convert_ids_to_tokens(token_id)
        neighbors = ft.get_nearest_neighbors(token)
        selected_neighbor_indices = []
        selected_neighbor_scores = []
        for (score, neigh) in neighbors:
            if neigh in vocab and vocab[neigh] in token_ids_kept:
                selected_neighbor_indices.append(token_ids_kept.index(vocab[neigh]))
                selected_neighbor_scores.append(score * score_discount)
            if len(selected_neighbor_indices) == topk_neigh:
                break
        # print(token, selected_neighbor_indices)
        # add the average of similarity_score * activation of each of the topk neighbor tokens to the current neuron representation
        if len(selected_neighbor_indices) > 0:
            all_layer_repr_aug[:, i] += torch.mean(all_layer_repr[:, selected_neighbor_indices] * torch.tensor(selected_neighbor_scores).unsqueeze(0), dim=1)

    # save the augmented neuron representations to file
    save_path = os.path.join(NEURON_REPR_DIR, 'neuron_repr_augmented.json')

    all_layer_repr_aug = all_layer_repr_aug.tolist()
    with open(save_path, 'w') as f:
        json.dump(all_layer_repr_aug, f)



def main(token_count_only=False):
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
    get_neuron_representations(model, tokenizer, config, dataset, device, token_count_only=token_count_only)


if __name__ == "__main__":
    # main(token_count_only=True)
    # filter_neuron_repr_with_token_count(min_count=400)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    augment_neuron_repr_with_token_similarity(tokenizer, topk_neigh=5, score_discount=0.5)


    # # load neuron representations
    # with open(f'{NEURON_REPR_DIR}/token_ids_to_keep.json', 'r') as f:
    #     token_ids_kept = json.load(f)
    
    # # print kept vocab
    # kept_vocab = tokenizer.convert_ids_to_tokens(token_ids_kept)
    # print(kept_vocab)