"""Factory for creating KeySonic models."""

from torch import nn

from keysonic.models.mlp import MLPHead


def build_model(
    model_name: str,
    *,
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> nn.Module:
    """Create a model from its name and dimensions.

    Args:
        model_name: Name of the model to create. Currently supports "mlp".
        input_size: Number of input features per sample.
        hidden_size: Number of neurons in the hidden layer.
        output_size: Number of output features or classes.

    Returns:
        An initialized model with trainable parameters.

    Raises:
        ValueError: If the model name is unsupported or dimensions
            are less than one.
    """
    name = model_name.strip().lower()

    if name == "mlp":
        return MLPHead(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=output_size,
        )

    raise ValueError(
        f"Unknown model: {model_name!r}. Available models: 'mlp'."
    )