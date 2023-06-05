from cluster import compute_clusters
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, BertConfig, BertModel, BertForPreTraining, BertForMaskedLM
from datasets import load_dataset
from constants import BATCH_SIZE, NUM_LAYERS, HIDDEN_DIM


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
    model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config).to(device)
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
    # set [PAD] tokens to be -100
    labels[labels == tokenizer.pad_token_id] = -100
    # randomly mask 15% of the input tokens to be [MASK]
    masked_indices = torch.bernoulli(torch.full(input_ids.size(), 0.15)).bool().to(device)
    # only mask those tokens that are not [PAD], [CLS], or [SEP]
    masked_indices = masked_indices & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
    input_ids[masked_indices] = tokenizer.mask_token_id
    # assert input_ids is different from labels
    assert not torch.equal(input_ids, labels)
    # set labels to -100 (ignore index) except the masked indices
    labels[~masked_indices] = -100

    with torch.no_grad():
        extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size()).to(device)
        hidden_states= model.bert.embeddings(input_ids=input_ids)
        hidden_states[:, :, []] = 0 # turn none of them off 
        for layer_id in range(1, NUM_LAYERS):
            hidden_states = model.bert.encoder.layer[layer_id - 1](hidden_states, attention_mask=extended_attention_mask)[0]
            hidden_states[:, :, []] = 0 # turn none of them off 
        layer_by_layer_hidden_states = hidden_states
        sequence_output = hidden_states # shape: (batch_size, seq_len, hidden_size)
        layer_by_layer_prediction_scores = model.cls(sequence_output) # shape: (batch_size, sequence_length, vocab_size)
        # compute MLM loss
        layer_by_layer_masked_lm_loss = loss_fct(layer_by_layer_prediction_scores.view(-1, config.vocab_size), labels.view(-1))
    
    with torch.no_grad():
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoder_last_hidden_states = outputs[0]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        prediction_logits = outputs.logits
        masked_lm_loss = outputs.loss
    
    # assert layer_by_layer_hidden_states == encoder_last_hidden_states
    print("layer by layer is equal: ", torch.all(torch.eq(layer_by_layer_hidden_states, encoder_last_hidden_states)))

    # assert layer_by_layer_prediction_scores == prediction_logits
    print("layer by layer prediction scores is equal: ", torch.all(torch.eq(layer_by_layer_prediction_scores, prediction_logits)))

    # assert layer_by_layer_masked_lm_loss == masked_lm_loss
    print("layer by layer masked lm loss is equal: ", torch.all(torch.eq(layer_by_layer_masked_lm_loss, masked_lm_loss)))

    # print masked_lm_loss
    print("masked_lm_loss: ", masked_lm_loss)
    print("layer_by_layer_masked_lm_loss: ", layer_by_layer_masked_lm_loss)


def test_turn_off_neurons():
    # load model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained('tokenizer_info')
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config).to(device)
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
    # set [PAD] tokens to be -100
    labels[labels == tokenizer.pad_token_id] = -100
    # randomly mask 15% of the input tokens to be [MASK]
    masked_indices = torch.bernoulli(torch.full(input_ids.size(), 0.15)).bool().to(device)
    # only mask those tokens that are not [PAD], [CLS], or [SEP]
    masked_indices = masked_indices & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
    input_ids[masked_indices] = tokenizer.mask_token_id
    # assert input_ids is different from labels
    assert not torch.equal(input_ids, labels)
    # set labels to -100 (ignore index) except the masked indices
    labels[~masked_indices] = -100

    # mlm loss should increase as we turn off more neurons
    for num_neurons_to_turn_off in [10, 20, 50, 100, 200, 400]:
        with torch.no_grad():
            extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.size()).to(device)
            hidden_states= model.bert.embeddings(input_ids=input_ids)
            neuron_indices_to_turn_off = torch.randperm(HIDDEN_DIM)[:num_neurons_to_turn_off].tolist()
            hidden_states[:, :, neuron_indices_to_turn_off] = 0 # turn off 100 neurons in each layer
            for layer_id in range(NUM_LAYERS - 1):
                hidden_states = model.bert.encoder.layer[layer_id](hidden_states, attention_mask=extended_attention_mask)[0]
                neuron_indices_to_turn_off = torch.randperm(HIDDEN_DIM)[:num_neurons_to_turn_off].tolist()
                hidden_states[:, :, neuron_indices_to_turn_off] = 0 # turn off 100 neurons in each layer
            hidden_states = hidden_states
            sequence_output = hidden_states # shape: (batch_size, seq_len, hidden_size)
            prediction_scores = model.cls(sequence_output) # shape: (batch_size, sequence_length, vocab_size)
            # compute MLM loss
            masked_lm_loss_turned_off = loss_fct(prediction_scores.view(-1, config.vocab_size), labels.view(-1))
            print(f"num_neurons_to_turn_off: {num_neurons_to_turn_off} masked_lm_loss_turned_off: {masked_lm_loss_turned_off}" )

def test_mask_tokens():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    top_tokens = ["apology", "sorry"]
    input_ids = torch.zeros((1, 10), dtype=torch.long)
    token_ids = tokenizer.convert_tokens_to_ids(top_tokens)
    input_ids[0, 5] = token_ids[0]
    input_ids[0, 6] = token_ids[1]
    print(token_ids)
    print(input_ids)
    masked_indices = torch.zeros(input_ids.size(), dtype=torch.bool)
    for token_id in token_ids:
        masked_indices = masked_indices | (input_ids == token_id)
    input_ids[masked_indices] = tokenizer.mask_token_id
    print(input_ids)
    assert input_ids[0, 5] == tokenizer.mask_token_id
    assert input_ids[0, 6] == tokenizer.mask_token_id
    labels = input_ids.clone()
    labels[~masked_indices] = -100
    print(labels)
    assert labels[0, 5] == tokenizer.mask_token_id
    assert labels[0, 6] == tokenizer.mask_token_id
    assert labels[0, 0] == -100
    assert labels[0, 1] == -100

if __name__ == '__main__':
    # test_compute_clusters()
    test_layer_by_layer_equal()
    # test_turn_off_neurons()
    # test_mask_tokens()