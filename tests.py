from cluster import compute_clusters
import torch
from transformers import AutoTokenizer, BertConfig, BertModel
from datasets import load_dataset
from constants import BATCH_SIZE, NUM_LAYERS


def test_compute_clusters():
    repr = torch.tensor([[1.4, 0], [0, 11], [0, 1]])  # first neuron activates on the 1st token, second neuron activates on the 2nd & 3rd token
    repr1 = torch.tensor([[1, 0], [0, 11], [0, 2]])  # first neuron activates on the 1st token, second neuron activates on the 2nd & 3rd token
    all_layer_repr = {0: repr.t(), 1: repr1.t()}
    # combine all layers
    all_layer_repr = torch.cat([all_layer_repr[i] for i in range(2)], dim=0)
    print(all_layer_repr.shape)  # (4, 3)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    compute_clusters(all_layer_repr, tokenizer, num_clusters=2, num_top_tokens=2)

def test_layer_by_layer_equal():
    # load model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained('tokenizer_info')
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = BertModel.from_pretrained("bert-base-uncased", config=config).to(device)
    print("Model loaded")

    # load data
    yelp = load_dataset("yelp_review_full")
    dataset = yelp["test"]["text"][:10000] # for development purpose, only use the first 10000 examples in yelp["test"]["text"]
    print("Dataset loaded")

    batch = dataset[:BATCH_SIZE]
    with torch.no_grad():
        temp = tokenizer.batch_encode_plus(
                    batch, add_special_tokens=True, padding=True, truncation=True, max_length=config.max_position_embeddings, return_tensors='pt', return_attention_mask=True)
        input_ids, attention_mask = temp["input_ids"].to(device), temp["attention_mask"].to(device)
        extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size()).to(device)
        hidden_states= model.embeddings(input_ids=input_ids)
        for layer_id in range(NUM_LAYERS - 1):
            hidden_states = model.encoder.layer[layer_id](hidden_states, attention_mask=extended_attention_mask)[0]
        layer_by_layer_hidden_states = hidden_states
    
    with torch.no_grad():
        temp = tokenizer.batch_encode_plus(
                    batch, add_special_tokens=True, padding=True, truncation=True, max_length=config.max_position_embeddings, return_tensors='pt', return_attention_mask=True)
        input_ids, attention_mask = temp["input_ids"].to(device), temp["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[2]
        print("len(all_hidden_states): ", len(all_hidden_states))
        encoder_last_hidden_states = all_hidden_states[-1]
    
    # assert layer_by_layer_hidden_states == encoder_last_hidden_states
    print("layer by layer is equal: ", torch.all(torch.eq(layer_by_layer_hidden_states, encoder_last_hidden_states)))


if __name__ == '__main__':
    test_compute_clusters()
    test_layer_by_layer_equal()