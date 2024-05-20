import torch
from torch import nn

class IntegrationMethodApplier(nn.Module):
    def __init__(self, integration_method="addition"):
        """
        Initialize the IntegrationMethodApplier with a specified integration method.
        
        Args:
            integration_method (str): The method of integration to use ("addition" or "multiplication").
        """
        super(IntegrationMethodApplier, self).__init__()
        self.integration_method = integration_method

    def forward(self, input_tensor, additional_input_vector):
        """
        Apply the integration method to the input tensor and additional input vector.
        
        Args:
            input_tensor (torch.Tensor): The original input tensor.
            additional_input_vector (torch.Tensor): The additional input vector to integrate.
        
        Returns:
            torch.Tensor: The tensor resulting from the integration.
        """
        if self.integration_method == "addition":
            return input_tensor + additional_input_vector
        elif self.integration_method == "multiplication":
            return input_tensor * additional_input_vector
        else:
            raise ValueError("Unsupported integration method")