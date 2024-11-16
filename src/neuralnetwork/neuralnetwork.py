import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class CarAIModel(nn.Module):
    def __init__(self, input_size):
        super(CarAIModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # Two outputs: throttle and steering

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Outputs are normalized between -1 and 1
        return x

# Initialize the model
input_size = len(input_features)
model = CarAIModel(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
