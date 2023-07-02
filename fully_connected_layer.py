import torch
import torch.nn as nn
import matplotlib.pyplot as plt

input_size = 4
output_size = 2
# Fully connected layer
fc_layer = nn.Linear(input_size, output_size)
# Weight matrix
w = fc_layer.weight.detach().numpy()

# Weight matrix visualization
plt.figure(figsize=(10, 6))
plt.imshow(w, cmap="coolwarm", aspect="auto")
plt.xlabel("Input Features")
plt.ylabel("Output units")
plt.title("Fully connected layer weights")
plt.colorbar()
plt.show()
