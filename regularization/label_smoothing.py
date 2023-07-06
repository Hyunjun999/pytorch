import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing) -> None:
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):  # pred = prediction, target = true value
        # One-hot encoded tensor
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (
            self.num_classes - 1
        )
        loss = torch.sum(-smooth_label * torch.log_softmax(pred, dim=1), dim=1)
        return torch.mean(loss)


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )
        # self.conv1 == nn.Conv2d(1, 32, kernel_size=3)
        # self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(kernel_size=2)

        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)

        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(64 * 5 * 5, 128)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


train_dataset = torchvision.datasets.FashionMNIST(
    root="./Fashion_MNIST", train=True, download=True, transform=ToTensor()
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

model = CustomModel()
# Hyperparams
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_classes = 10
smoothing = 0.2
criterion = LabelSmoothingLoss(num_classes, smoothing)
train_loss_without_smoothing = []
train_loss_with_smoothing = []
epochs = 20

# Training
model.train()
for epoch in range(epochs):
    without_smoothing = 0.0
    with_smoothing = 0.0
    for img, label in train_loader:
        opt.zero_grad()
        # Without label smoothing case
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        opt.step()
        without_smoothing += loss.item()

        # With label smoothing case
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        opt.step()
        with_smoothing += loss.item()

    # Append loss for every epoch
    train_loss_without_smoothing.append(without_smoothing / len(train_loader))
    train_loss_with_smoothing.append(with_smoothing / len(train_loader))
    print(f"{epoch + 1}th epoch")

plt.plot(
    range(1, epochs + 1), train_loss_without_smoothing, label="W/o label smoothing"
)
plt.plot(range(1, epochs + 1), train_loss_with_smoothing, label="W/ label smoothing")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training loss")
plt.legend()
plt.show()
