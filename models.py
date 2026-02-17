"""MLP model with configurable depth, hidden widths, and dropout."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP with arbitrary depth and configurable hidden layer widths.

    Architecture: input_dim -> [hidden (ReLU, Dropout)] x N -> 10

    The linear layers are stored in self.layers as an nn.ModuleList.
    layers[0] is input->hidden1, layers[1] is hidden1->hidden2, ...,
    layers[-1] is hiddenN->10.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_widths: list[int] | None = None,
        dropout: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        if hidden_widths is None:
            hidden_widths = [256, 256, 256, 256]

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self._num_classes = num_classes

        # Build layers: input -> h1 -> h2 -> ... -> hN -> num_classes
        dims = [input_dim] + hidden_widths + [num_classes]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        # All layers except the last get ReLU + Dropout
        for layer in self.layers[:-1]:
            x = self.dropout_layer(self.relu(layer(x)))
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

    @property
    def input_dim(self) -> int:
        return self.layers[0].in_features

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers) - 1

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        """Return current hidden layer widths."""
        return tuple(layer.out_features for layer in self.layers[:-1])
