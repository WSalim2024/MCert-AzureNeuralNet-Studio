import torch
import torch.nn as nn
import torch.nn.functional as F


# --- ORIGINAL SIMPLE NN (For MNIST/Fashion) ---
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- NEW CNN ARCHITECTURE (For CIFAR-10) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 channels (RGB), 32x32 images
        # Conv Layer 1: 3 in, 32 out
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Conv Layer 2: 32 in, 64 out
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully Connected Layers
        # Image reduces to 8x8 after two poolings (32->16->8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.conv1(x)))
        # Layer 2
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        # Dense
        x = F.relu(self.fc1(x))
        return self.fc2(x)