import torch
import torch.optim as optim

# 1. Setup the basic components
inputs = torch.tensor([1.5, 2.0, -0.5])
target = torch.tensor([1.0])  # The correct answer we want

# CRITICAL: We add requires_grad=True so PyTorch tracks how to adjust these numbers
weights = torch.tensor([0.8, -0.2, 0.5], requires_grad=True)
bias = torch.tensor([0.1], requires_grad=True)

# 2. Choose an Optimizer (The mechanic that updates the weights)
# SGD means Stochastic Gradient Descent. lr=0.1 is the step size (learning rate)
optimizer = optim.SGD([weights, bias], lr=0.1)

print("--- Training Agastya over 5 steps ---")

for epoch in range(5):
    # Clear out old gradients from the previous turn
    optimizer.zero_grad()
    
    # Forward Pass: Make a guess
    summation = torch.dot(inputs, weights) + bias
    output = torch.relu(summation)
    
    # Calculate the Loss (Mean Squared Error)
    loss = (output - target) ** 2
    
    # Backpropagation: PyTorch calculates the calculus gradients automatically!
    loss.backward()
    
    # Update the weights and bias based on the gradients
    optimizer.step()
    
    print(f"Step {epoch+1}: Prediction = {output.item():.4f} | Loss = {loss.item():.4f}")

print("\n--- Optimized Values ---")
print(f"New Weights: {weights.detach().numpy()}")
print(f"New Bias: {bias.item():.4f}")