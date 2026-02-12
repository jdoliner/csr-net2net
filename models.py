"""MLP model with configurable hidden widths for MNIST classification."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP for MNIST with configurable hidden layer widths.

    Architecture: 784 -> hidden1 (ReLU) -> hidden2 (ReLU) -> 10
    """

    def __init__(self, hidden1: int = 128, hidden2: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten to [B, 784]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @property
    def hidden_widths(self) -> tuple[int, int]:
        """Return current hidden layer widths."""
        return self.fc1.out_features, self.fc2.out_features
