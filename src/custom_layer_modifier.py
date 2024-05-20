from torch import nn

class CustomLayerBERTModifier(nn.Module):
    def __init__(self, model, layer_number, integration_method_applier):

        super(CustomLayerBERTModifier, self).__init__()
        self.bert = model
        self.layer_number = layer_number
        self.integration_method_applier = integration_method_applier

    def forward(self, input_ids, attention_mask, additional_input_vector):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        modified_layer_input = self.integration_method_applier(hidden_states[self.layer_number - 1], additional_input_vector)
        for i in range(self.layer_number, len(self.bert.encoder.layer)):
            modified_layer_input = self.bert.encoder.layer[i](modified_layer_input)[0]

        return modified_layer_input
