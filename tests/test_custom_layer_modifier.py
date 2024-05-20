import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.setup import prepare_inputs, setup_model_and_tokenizer, create_additional_input_vector
from src.integration import IntegrationMethodApplier
from src.custom_layer_modifier import CustomLayerBERTModifier

def test_custom_layer_modifier(input_text, layer_number, integration_method):
    model, tokenizer = setup_model_and_tokenizer()

    additional_input_vector = create_additional_input_vector(model.config.hidden_size)

    integration_method_applier = IntegrationMethodApplier(integration_method)
    custom_model = CustomLayerBERTModifier(model, layer_number, integration_method_applier)

    inputs = prepare_inputs(input_text, tokenizer)

    with torch.no_grad():
        outputs = custom_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], additional_input_vector=additional_input_vector)

    print("Output shape: ", outputs.shape)
    return outputs


if __name__ == "__main__":
    test_custom_layer_modifier("Hello, how are you?", 10, "addition")