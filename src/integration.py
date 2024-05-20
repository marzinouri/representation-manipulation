from torch import nn

class IntegrationMethodApplier(nn.Module):
    def __init__(self, integration_method="addition"):
        super(IntegrationMethodApplier, self).__init__()
        self.integration_method = integration_method

    def forward(self, input_tensor, additional_input_vector):
        if self.integration_method == "addition":
            return input_tensor + additional_input_vector
        elif self.integration_method == "multiplication":
            return input_tensor * additional_input_vector
        else:
            raise ValueError("Unsupported integration method")
