import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """
    A simple Feedforward Neural Network for MNIST classification.
    Input: 784 features (28x28 pixels)
    Hidden: 128 neurons (ReLU activation)
    Output: 10 classes (Digits 0-9)
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input Layer
        self.fc2 = nn.Linear(hidden_size, num_classes) # Output Layer

    def forward(self, x):
        # Flatten image input
        x = x.view(-1, 28 * 28)
        # Activation function
        x = F.relu(self.fc1(x)) # ReLU Activation
        x = self.fc2(x)
        return x