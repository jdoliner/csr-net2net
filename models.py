"""MLP model with configurable input dimension and hidden widths."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP with configurable input dimension and hidden layer widths.

    Architecture: input_dim -> hidden1 (ReLU) -> hidden2 (ReLU) -> 10
    """

    def __init__(self, input_dim: int = 3072, hidden1: int = 256, hidden2: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten to [B, input_dim]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
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
