import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import BertModel
from src.setup import prepare_inputs, setup_model_and_tokenizer, create_additional_input_vector
from src.integration import IntegrationMethodApplier
from src.hook_based_modifier import HookBasedBERTModifier

def test_hook_based_modifier(input_text, layer_number, integration_method):
    model, tokenizer = setup_model_and_tokenizer()
    additional_input_vector = create_additional_input_vector(model.config.hidden_size)
    integration_method_applier = IntegrationMethodApplier(integration_method)
    
    modifier = HookBasedBERTModifier(model, layer_number, integration_method_applier)
   
    inputs = prepare_inputs(input_text, tokenizer)

    with torch.no_grad():
        outputs_without_hook = model(**inputs)

    modifier.register_hook(additional_input_vector)

    with torch.no_grad():
        outputs_with_hook = model(**inputs)

    modifier.remove_hook()

    output_difference = torch.abs(outputs_with_hook.last_hidden_state - outputs_without_hook.last_hidden_state)
    print("Output difference: ", torch.sum(output_difference).item())
    return outputs_without_hook, outputs_with_hook
  
if __name__ == "__main__":
    test_hook_based_modifier("Hello, how are you?", 10, "addition")