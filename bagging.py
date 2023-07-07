import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bagging(Bootstrapping aggregating) is one way of Ensemble which can be considered as a parallel learning


# CNN model
class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Load CIFAR10 dataset
train_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./CIFAR10", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./CIFAR10/", train=False, download=True, transform=test_transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create an ensemble of CNN models
num_models = 5
models = [CNN().to(device) for _ in range(num_models)]

# Hyperparams
criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

# Training loop
epochs = 4
for e in range(epochs):  # Larger epochs may bring the larger accuracy
    print(f"Trainig at {e}th epoch")
    for model, opt in zip(models, optimizers):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

    # Evaluation for every epoch
    for model in models:
        model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            ensemble_outputs = torch.zeros((images.size(0), 10)).to(
                device
            )  # Initialize w/ appropriate size
            for model in models:
                outputs = model(images)
                ensemble_outputs += outputs / num_models

            _, predicted = torch.max(ensemble_outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, predictions)
    print(predictions)  # it would be like [5, 1, 2, 8, 3, 0, 9...]
    print(f"Epoch {e + 1} / {epochs}, Acc : {acc}")
