import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import BertModel
from src.setup import prepare_inputs, setup_model_and_tokenizer, create_additional_input_vector
from src.integration import IntegrationMethodApplier
from src.hook_based_modifier import HookBasedBERTModifier

def test_hook_based_modifier(input_text, layer_number, integration_method):
    """
    Test the HookBasedBERTModifier by comparing model outputs with and without the hook.
    
    Parameters:
    input_text (str): The input text to process with the BERT model.
    layer_number (int): The layer number where the hook will be applied.
    integration_method (str): The method used for integrating the additional input ('addition' or 'multiplication').
    """

    # Setup the model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Create an additional input vector
    additional_input_vector = create_additional_input_vector(model.config.hidden_size)

    # Initialize the integration method applier
    integration_method_applier = IntegrationMethodApplier(integration_method)

    # Initialize the modifier with the model, layer number, and integration method
    modifier = HookBasedBERTModifier(model, layer_number, integration_method_applier)

    # Prepare the inputs for the model
    inputs = prepare_inputs(input_text, tokenizer)

    # Get model outputs without the hook
    with torch.no_grad():
        outputs_without_hook = model(**inputs)
    
    # Register the hook with the additional input vecto
    modifier.register_hook(additional_input_vector)

    # Get model outputs with the hook
    with torch.no_grad():
        outputs_with_hook = model(**inputs)

    # Remove the hook
    modifier.remove_hook()

    # Calculate the difference between the outputs with and without the hook
    output_difference = torch.abs(outputs_with_hook.last_hidden_state - outputs_without_hook.last_hidden_state)
    print("Output difference: ", torch.sum(output_difference).item())
    # return outputs_without_hook, outputs_with_hook
  
if __name__ == "__main__":
    test_hook_based_modifier("Hello, how are you?", 10, "addition")
