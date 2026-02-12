"""MLP model with configurable input dimension, hidden widths, and dropout."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP with configurable input dimension, hidden layer widths, and dropout.

    Architecture: input_dim -> hidden1 (ReLU, Dropout) -> hidden2 (ReLU, Dropout) -> 10
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden1: int = 256,
        hidden2: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten to [B, input_dim]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self.fc1.in_features

    @property
    def hidden_widths(self) -> tuple[int, int]:
        """Return current hidden layer widths."""
        return self.fc1.out_features, self.fc2.out_features
