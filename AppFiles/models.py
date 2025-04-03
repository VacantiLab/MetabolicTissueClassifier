import torch
import torch.nn as nn

# Define the fully connected neural network
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Output layer
        #self.softmax = nn.Softmax(dim=1)  # Softmax activation for output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        x = self.fc3(x)  # Output layer (logits)
        #x = self.softmax(x)  # Apply softmax to get probabilities
        return x