import torch

# Step 1: Agastya's Inputs (Data)
# Let's pretend we are feeding Agastya 3 pieces of information (e.g., temperature, humidity, wind speed)
inputs = torch.tensor([1.5, 2.0, -0.5])

# Step 2: Agastya's Weights (Randomly initialized at "birth")
# Each input gets its own weight determining its importance
weights = torch.tensor([0.8, -0.2, 0.5])

# Step 3: Agastya's Bias 
# A single number to adjust the threshold
bias = torch.tensor([0.1])

# Step 4: The Core Math (Summation)
# Multiply inputs by weights, and add the bias
# (1.5 * 0.8) + (2.0 * -0.2) + (-0.5 * 0.5) + 0.1
summation = torch.dot(inputs, weights) + bias

# Step 5: The Activation Function (The "Decision" Filter)
# We use ReLU (Rectified Linear Unit). If the number is negative, it becomes 0. If positive, it stays the same.
output = torch.relu(summation)

print("--- Agastya's First Thought ---")
print(f"Inputs: {inputs}")
print(f"Raw Summation: {summation.item():.4f}")
print(f"Final Neuron Output: {output.item():.4f}")