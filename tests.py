from cluster import compute_clusters
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, BertConfig, BertModel, BertForPreTraining
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
    model = BertForPreTraining.from_pretrained("bert-base-uncased", config=config).to(device)
    print("Model loaded")

    # define loss function
    loss_fct = CrossEntropyLoss() # MLM

    # load data
    yelp = load_dataset("yelp_review_full")
    dataset = yelp["test"]["text"][:10000] # for development purpose, only use the first 10000 examples in yelp["test"]["text"]
    print("Dataset loaded")

    batch = dataset[:BATCH_SIZE]
    temp = tokenizer.batch_encode_plus(
                batch, add_special_tokens=True, padding=True, truncation=True, max_length=config.max_position_embeddings, return_tensors='pt', return_attention_mask=True)
    input_ids, attention_mask = temp["input_ids"].to(device), temp["attention_mask"].to(device)
    labels = input_ids.clone()
    # randomly mask 15% of the input tokens to be [MASK]
    masked_indices = torch.bernoulli(torch.full(input_ids.size(), 0.15)).bool().to(device)
    input_ids[masked_indices] = tokenizer.mask_token_id
        
    with torch.no_grad():
        extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size()).to(device)
        hidden_states= model.bert.embeddings(input_ids=input_ids)
        for layer_id in range(NUM_LAYERS - 1):
            hidden_states = model.bert.encoder.layer[layer_id](hidden_states, attention_mask=extended_attention_mask)[0]
        layer_by_layer_hidden_states = hidden_states
        sequence_output = hidden_states # shape: (batch_size, seq_len, hidden_size)
        pooled_output = model.bert.pooler(hidden_states) # shape: (batch_size, hidden_size)
        layer_by_layer_prediction_scores, _ = model.cls(sequence_output, pooled_output) # shape: (batch_size, sequence_length, vocab_size)
        # compute MLM loss
        layer_by_layer_masked_lm_loss = loss_fct(layer_by_layer_prediction_scores.view(-1, config.vocab_size), labels.view(-1))
    
    with torch.no_grad():
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[2]
        print("len(all_hidden_states): ", len(all_hidden_states))
        encoder_last_hidden_states = all_hidden_states[-1]

        # get prediction scores
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction_logits = outputs.prediction_logits
        # compute MLM loss
        masked_lm_loss = loss_fct(prediction_logits.view(-1, config.vocab_size), labels.view(-1))
    
    # assert layer_by_layer_hidden_states == encoder_last_hidden_states
    print("layer by layer is equal: ", torch.all(torch.eq(layer_by_layer_hidden_states, encoder_last_hidden_states)))

    # assert layer_by_layer_prediction_scores == prediction_logits
    print("layer by layer prediction scores is equal: ", torch.all(torch.eq(layer_by_layer_prediction_scores, prediction_logits)))

    # assert layer_by_layer_masked_lm_loss == masked_lm_loss
    print("layer by layer masked lm loss is equal: ", torch.all(torch.eq(layer_by_layer_masked_lm_loss, masked_lm_loss)))


if __name__ == '__main__':
    # test_compute_clusters()
    test_layer_by_layer_equal()