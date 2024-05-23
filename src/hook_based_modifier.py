class HookBasedBERTModifier:
    def __init__(self, model, layer_number, integration_method_applier):
        """
        Initialize the HookBasedBERTModifier.
        
        Parameters:
        model (BertModel): The BERT model to modify.
        layer_number (int): The layer number to apply the modification.
        integration_method_applier (IntegrationMethodApplier): The method to integrate the additional input.
        """
        self.model = model
        self.layer_number = layer_number
        self.integration_method_applier = integration_method_applier
        self.hook = None

    def modify_output(self, module, input, output):
        """
        Modify the output of the specified layer.
        
        Parameters:
        module (nn.Module): The module to which the hook is attached.
        input (tuple): The input to the module.
        output (torch.Tensor): The output from the module.
        
        Returns:
        tuple: The modified output.
        """
        input_tensor = input[0]
        modified_output = self.integration_method_applier(input_tensor, self.additional_input_vector)
        return (modified_output,)

    def register_hook(self, additional_input_vector):
        """
        Register a hook on the specified layer to modify its output.
        
        Parameters:
        additional_input_vector (torch.Tensor): The vector to integrate with the layer output.
        """
        self.additional_input_vector = additional_input_vector
        layer = self.model.encoder.layer[self.layer_number - 1]
        self.hook = layer.register_forward_hook(self.modify_output)

    def remove_hook(self):
        """
        Remove the registered hook if it exists.
        """
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
