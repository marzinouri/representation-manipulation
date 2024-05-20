import torch
from torch import nn

class BERTModifier:
    def __init__(self, model, layer_number, integration_method_applier):
        """
        Initialize the BERTModifier with the BERT model, target layer number, and integration method applier.
        
        Args:
            model (BertModel): The BERT model to modify.
            layer_number (int): The layer number where the modification is to be applied.
            integration_method_applier (IntegrationMethodApplier): The integration method applier instance.
        """
        self.model = model
        self.layer_number = layer_number
        self.integration_method_applier = integration_method_applier
        self.hook = None

    def hook_fn(self, module, input, output):
        """
        Hook function to modify the output of a specific BERT layer.
        
        Args:
            module (nn.Module): The layer module.
            input (tuple): The input to the layer.
            output (torch.Tensor): The original output of the layer.
        
        Returns:
            torch.Tensor: The modified output.
        """
        additional_input_vector = self.additional_input_vector
        modified_output = self.integration_method_applier(output, additional_input_vector)
        return modified_output

    def add_hook(self, additional_input_vector):
        """
        Add a forward hook to the specified BERT layer.
        
        Args:
            additional_input_vector (torch.Tensor): The additional input vector for modification.
        """
        self.additional_input_vector = additional_input_vector
        target_layer = self.model.encoder.layer[self.layer_number].output
        self.hook = target_layer.register_forward_hook(self.hook_fn)

    def remove_hook(self):
        """Remove the forward hook from the BERT layer."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
