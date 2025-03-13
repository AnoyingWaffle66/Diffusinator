# import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, num):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4 * num * num,  8 * num),
            nn.ReLU(),
            nn.Linear(num * 8, num * 16),
            nn.ReLU(),
            nn.Linear(num * 16, 4 * num * num)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits