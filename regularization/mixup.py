import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)),
    ]
)
train_dataset = datasets.CIFAR10(
    root="./CIFAR10", train=True, transform=train_transform, download=True
)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)


def mixup_data(x, y, alpha=1.0):
    batch_size = x.size()[0]  # input data * batch size
    lam = torch.rand(batch_size, 1, 1, 1)  # Random mixup values btw 0 and 1
    lam = torch.max(lam, 1 - lam)  # Maintain min value as 0.5

    mixed_x = lam * x + (1 - lam) * x.flip(dims=[0, 2, 3])
    indices = torch.randperm(batch_size)
    mixed_y = lam.squeeze() * y + (1 - lam.squeeze()) * y[indices]
    mixed_y = mixed_y.type(torch.long)

    return mixed_x, mixed_y


def plot_img(image, label, title):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    label = label.numpy()
    for i, ax in enumerate(axes.flat):
        img = image[i].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label:{label[i]}")
        ax.axis("off")
    plt.show()


# Hyperparameter
model = CustomModel()
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.01)
epochs = 10
train_loss_without_mixup_l = []
train_loss_with_mixup_l = []

# Training

idx = 0
for epoch in range(epochs):
    train_loss_without_mixup = 0.0
    train_loss_with_mixup = 0.0

    for inputs, labels in train_dataloader:
        opt.zero_grad()
        imges, labels = mixup_data(inputs, labels)

        mixed_images = inputs.cpu().numpy()
        mixed_images = np.transpose(mixed_images, (0, 2, 3, 1))
        mixed_images = np.squeeze(mixed_images)

        if idx == 0:
            plot_img(mixed_images, labels.squeeze(), "Mixed Image with label smoothing")
            idx = 1

        outputs_without_mixup = model(inputs)
        outputs_with_mixup = model(imges)
        labels = torch.squeeze(labels)
        loss_without_mixup = criterion(outputs_without_mixup, labels)
        loss_with_mixup = criterion(outputs_with_mixup, labels)

        loss_without_mixup.backward()
        loss_with_mixup.backward()
        opt.step()
        train_loss_without_mixup += loss_without_mixup.item()
        train_loss_with_mixup += loss_with_mixup.item()

    train_loss_without_mixup_l.append(train_loss_without_mixup / len(train_dataloader))
    train_loss_with_mixup_l.append(train_loss_with_mixup / len(train_dataloader))

print("Without mixup:", train_loss_without_mixup_l)
print("With mixup", train_loss_with_mixup_l)
plt.plot(range(1, epochs + 1), train_loss_without_mixup_l, label="W/o mixup loss")
plt.plot(range(1, epochs + 1), train_loss_with_mixup_l, label="W mixup loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.show()
