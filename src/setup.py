import torch
from transformers import BertModel, BertTokenizer

def set_seed():
    # Set the random seed for reproducibility.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_model_and_tokenizer(model_name='bert-base-uncased'):
    # Load a pre-trained BERT model and tokenizer.
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_inputs(text, tokenizer):
    # Tokenize input text and convert to tensors.
    return tokenizer(text, return_tensors='pt')

def create_additional_input_vector(hidden_size):
    # Create a random tensor of specified hidden size.
    set_seed()
    return torch.randn(1, 1, hidden_size)
