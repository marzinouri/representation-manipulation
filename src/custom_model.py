import torch
from torch import nn

class CustomBERTModel(nn.Module):
    def __init__(self, model, layer_number, integration_method_applier):
        """
        Initialize the CustomBERTModel with the BERT model, target layer number, and integration method applier.
        
        Args:
            model (BertModel): The original BERT model to modify.
            layer_number (int): The layer number where the modification is to be applied.
            integration_method_applier (IntegrationMethodApplier): The integration method applier instance.
        """
        super(CustomBERTModel, self).__init__()
        self.bert = model
        self.layer_number = layer_number
        self.integration_method_applier = integration_method_applier

    def forward(self, input_ids, attention_mask, additional_input_vector):
        """
        Forward pass for the custom BERT model.
        
        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            additional_input_vector (torch.Tensor): Additional input vector for integration.
        
        Returns:
            torch.Tensor: The output tensor after modification.
        """
        # Get all hidden states
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Modify the specified layer's input
        modified_layer_input = self.integration_method_applier(hidden_states[self.layer_number - 1], additional_input_vector)

        # Recreate the rest of the BERT layers from the modified input
        for i in range(self.layer_number, len(self.bert.encoder.layer)):
            modified_layer_input = self.bert.encoder.layer[i](modified_layer_input)[0]

        return modified_layer_input
