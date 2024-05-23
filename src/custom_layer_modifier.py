from torch import nn

class CustomLayerBERTModifier(nn.Module):
    def __init__(self, model, layer_number, integration_method_applier):
        """
        Initialize the CustomLayerBERTModifier.
        
        Parameters:
        model (BertModel): The BERT model to modify.
        layer_number (int): The layer number to start the modification.
        integration_method_applier (IntegrationMethodApplier): The method to integrate the additional input.
        """
        super(CustomLayerBERTModifier, self).__init__()
        self.bert = model
        self.layer_number = layer_number
        self.integration_method_applier = integration_method_applier

    def forward(self, input_ids, attention_mask, additional_input_vector):
        """
        Forward pass to apply the custom layer modification.
        
        Parameters:
        input_ids (torch.Tensor): The input IDs for the BERT model.
        attention_mask (torch.Tensor): The attention mask for the BERT model.
        additional_input_vector (torch.Tensor): The vector to integrate with the layer output.
        
        Returns:
        torch.Tensor: The modified output after applying the integration method.
        """
        # Get the BERT model outputs with hidden states
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Apply the integration method to the specified layer's hidden states
        modified_layer_input = self.integration_method_applier(hidden_states[self.layer_number - 1], additional_input_vector)
        
        # Pass the modified input through the remaining layers
        for i in range(self.layer_number, len(self.bert.encoder.layer)):
            modified_layer_input = self.bert.encoder.layer[i](modified_layer_input)[0]

        return modified_layer_input

