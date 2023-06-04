import os
import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, BertModel, BertConfig
from datasets import load_dataset

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


def load_dataset_from_hf(dev=False):
    if dev:
        yelp = load_dataset("yelp_review_full")
        dataset = yelp["test"]["text"][:10000] # for development purpose, only use the first 10000 examples in yelp["test"]["text"]
    else:
        data_files = {"validation": "en/c4-validation.*.json.gz"}
        dataset = load_dataset("allenai/c4", data_files=data_files, split="validation")
        # dataset = load_dataset("c4", "en", split="validation")
        dataset = dataset["text"][:10000]  # TODO: use all data
    print("Dataset loaded")
    return dataset


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
    main()
