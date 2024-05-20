import torch
from transformers import BertModel
from src.setup import setup_model_and_tokenizer, create_additional_input_vector
from src.integration import IntegrationMethodApplier
from src.hook_based_modifier import HookBasedBERTModifier

def test_hook_based_modifier(input_text, layer_number, integration_method):
    model, tokenizer = setup_model_and_tokenizer()
    additional_input_vector = create_additional_input_vector(model.config.hidden_size)
    integration_method_applier = IntegrationMethodApplier(integration_method)
    
    modifier = HookBasedBERTModifier(model, layer_number, integration_method_applier)
   
    # Tokenize the input text
    inputs = prepare_inputs(input_text, tokenizer)

    # Run the model without the hook and save the output
    with torch.no_grad():
        outputs_without_hook = model(**inputs)

    # Register the hook
    modifier.register_hook(additional_input_vector)

    # Run the model with the hook and save the output
    with torch.no_grad():
        outputs_with_hook = model(**inputs)

    # Remove the hook
    modifier.remove_hook()
    inputs = tokenizer("Hello, how are you?", return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Register the hook
    modifier.register_hook(additional_input_vector)

    # Run the model with the hook and save the output
    with torch.no_grad():
        outputs_with_hook = model(**inputs)

    # Remove the hook
    modifier.remove_hook()

    # Compare the actual tensor values or any specific part of the output
    output_difference = torch.abs(outputs_with_hook.last_hidden_state - outputs_without_hook.last_hidden_state)
    print("Output difference: ", torch.sum(output_difference).item())
    return outputs_without_hook, outputs_with_hook
  
