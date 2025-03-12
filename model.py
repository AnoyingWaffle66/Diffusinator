# import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4 * 32 * 32)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits