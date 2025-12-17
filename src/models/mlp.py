import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout: float = 0.2):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        # final output layer → 1 logit
        layers.append(nn.Linear(prev, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # return logits (shape: [batch])
        return self.model(x).squeeze(1)
