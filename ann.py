import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fun1 = nn.Linear(
            input_size, hidden_size
        )  # Pass from the input to hidden layer
        self.relu = nn.ReLU()  # Activation fun at the hidden layer
        self.fun2 = nn.Linear(hidden_size, output_size)  # Calculate

    def forward(self, x):
        out = self.fun1(x)  # Input - hidden calculation
        out = self.relu(out)  # Apply activation w/ hidden
        out = self.fun2(out)  # Hidden - output calculation
        return out


input_size = 784
hidden_size = 256
output_size = 10
model = ANN(input_size, hidden_size, output_size)

# Loss function & hyperparams
criterion = nn.CrossEntropyLoss()
lr = 0.01
opt = torch.optim.SGD(model.parameters(), lr)

# Create test & label data
inputs = torch.randn(100, input_size)
labels = torch.randint(0, output_size, (100,))
print(inputs)
print(labels)

epochs = 10
for e in range(epochs):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (e + 1) % 1 == 0:
        print(f"Epoch = {e} / {epochs}, Loss = {loss.item()}")
