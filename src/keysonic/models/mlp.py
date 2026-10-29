"""Reusable MLP classification head for KeySonic models"""

import torch 
from torch import nn


class MLPHead(nn.Module):
    """Classify input features using a multilayer perceptron.

    Flattens all dimensions except the batch dimension before applying
    the linear layers. Can process raw matrices or CNN features.

    Args:
        input_size: Number of features per sample after flattening.
        hidden_size: Number of neurons in the hidden layer.
        num_classes: Number of output classes.

    Raises:
        ValueError: If any dimension is less than one.
    """

    def __init__(self,input_size:int,hidden_size:int,num_classes=int)->None:
        super().__init__()
        if min(input_size,hidden_size,num_classes)<1:
            raise ValueError("All dimension must be grater than zero")

        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,hidden_size),
            nn.Relu(),
            nn.Linear(hidden_size,num_classes)
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute raw class scores for a batch of samples.

        Args:
            inputs: Tensor with shape (batch_size, ...). Each sample
                must contain exactly input_size features.

        Returns:
            Logits with shape (batch_size, num_classes).
        """
        return self.layers(inputs)