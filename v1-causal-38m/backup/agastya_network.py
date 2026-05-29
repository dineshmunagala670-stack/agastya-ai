import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define Agastya's Multi-Layer Architecture
class AgastyaBrain(nn.Module):
    def __init__(self):
        super(AgastyaBrain, self).__init__()
        # First layer: Takes 3 inputs and connects them to 4 hidden neurons
        self.hidden_layer = nn.Linear(3, 4)
        # Second layer: Takes those 4 hidden neurons and connects them to 1 output neuron
        self.output_layer = nn.Linear(4, 1)
        # Activation function to introduce non-linear logic
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass inputs through the hidden layer, then apply the ReLU filter
        x = self.relu(self.hidden_layer(x))
        # Pass the result through the output layer to get the final score
        x = self.output_layer(x)
        return x

# Instantiate the network
model = AgastyaBrain()

# Step 2: Setup inputs and target (Notice the extra brackets [] for batch formatting)
inputs = torch.tensor([[1.5, 2.0, -0.5]])
target = torch.tensor([[1.0]])

# Step 3: Define Loss function and Optimizer
criterion = nn.MSELoss() # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.05) # Lower learning rate to avoid overshooting

print("--- Training Agastya's First Multi-Layer Network ---")

for epoch in range(5):
    optimizer.zero_grad()
    
    # Forward pass: Feed data through the whole network layout
    predictions = model(inputs)
    
    # Calculate overall error
    loss = criterion(predictions, target)
    
    # Backpropagation: Calculates gradients for all neurons across all layers instantly
    loss.backward()
    
    # Update all parameters
    optimizer.step()
    
    print(f"Step {epoch+1}: Prediction = {predictions.item():.4f} | Loss = {loss.item():.4f}")