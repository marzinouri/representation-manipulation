import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.setup import prepare_inputs, setup_model_and_tokenizer, create_additional_input_vector
from src.integration import IntegrationMethodApplier
from src.custom_layer_modifier import CustomLayerBERTModifier

def test_custom_layer_modifier(input_text, layer_number, integration_method):
    """
    Test the CustomLayerBERTModifier by comparing model outputs with the custom layer modification.
    
    Parameters:
    input_text (str): The input text to process with the BERT model.
    layer_number (int): The layer number to start the modification.
    integration_method (str): The method used for integrating the additional input ('addition' or 'multiplication').
    """
    # Setup the model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Create an additional input vector
    additional_input_vector = create_additional_input_vector(model.config.hidden_size)

    # Initialize the integration method applier
    integration_method_applier = IntegrationMethodApplier(integration_method)

    # Initialize the custom model with the integration method applier
    custom_model = CustomLayerBERTModifier(model, layer_number, integration_method_applier)

    # Prepare the inputs for the model
    inputs = prepare_inputs(input_text, tokenizer)

    # Get model outputs with the custom layer modification
    with torch.no_grad():
        outputs = custom_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], additional_input_vector=additional_input_vector)

    print("Output shape: ", outputs.shape)
    # return outputs


if __name__ == "__main__":
    test_custom_layer_modifier("Hello, how are you?", 10, "addition")
