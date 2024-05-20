class HookBasedBERTModifier:
    def __init__(self, model, layer_number, integration_method_applier):
        self.model = model
        self.layer_number = layer_number
        self.integration_method_applier = integration_method_applier
        self.hook = None

    def modify_input(self, module, input):
        input_tensor = input[0]
        modified_input = self.integration_method_applier(input_tensor, self.additional_input_vector)
        return (modified_input,)

    def register_hook(self, additional_input_vector):
        self.additional_input_vector = additional_input_vector
        layer = self.model.encoder.layer[self.layer_number - 1] 
        self.hook = layer.register_forward_pre_hook(self.modify_input)

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None