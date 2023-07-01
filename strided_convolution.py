import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Output size height and width = (input_h or input_w + 2 * padding - kernel_h or kernwl_w ) / 2 + 1

input_data = torch.randn(1, 1, 28, 28)  # batch size = 1, # of output = 1, 28 * 28 size
print(input_data.shape)
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)

output = conv(input_data)

print(output.shape)  # 1, 16, 14, 14 down-sampling

plt.subplot(1, 2, 1)
plt.imshow(input_data.squeeze(), cmap="gray")
plt.title("input")

plt.subplot(1, 2, 2)
plt.imshow(output.squeeze().detach().numpy()[0], cmap="gray")
plt.title("output")

plt.tight_layout()
plt.show()
