import torch
from transformers import BertModel, BertTokenizer

def setup_model_and_tokenizer(model_name='bert-base-uncased'):
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_inputs(text, tokenizer):
    return tokenizer(text, return_tensors='pt')

def create_additional_input_vector(hidden_size):
    return torch.randn(1, 1, hidden_size)
