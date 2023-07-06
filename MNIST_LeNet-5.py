import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

random_seed = 777
lr = 0.001
batch_size = 32
epochs = 20

img_size = 32
class_number = 10

trasnform = transforms.Compose(
    [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
)

train_dataset = datasets.MNIST(
    root="./MNIST_train/", train=True, transform=trasnform, download=True
)

valid_dataset = datasets.MNIST(
    root="./MNIST_test/", train=False, transform=trasnform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis("off")
    plt.imshow(train_dataset.data[index], cmap="gray_r")
fig.suptitle("MNIST DATASET")
plt.show()


# LeNet5 definition
class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            # (32, 6, h, w)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),  # (32, 6, h / 2, w / 2)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # (32, 16, h / 2, w / 2)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            ## (32, 16, h / 4, w / 4)
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            ## (32, 120, h / 4, w / 4)
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=class_number),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)

        return logits, probs


def acc(model, data_loader, device):
    correct_pred, n = 0, 0

    # Evaluation mode
    model.eval()

    with torch.no_grad():
        for x, y_true in data_loader:
            x = x.to(device)
            y = y_true.to(device)

            _, y_prob = model(x)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y.size(0)
            correct_pred += (predicted_labels == y).sum()

    return correct_pred.float() / n


def train(train_loader, model, criterion, optimizer, device):
    model.train()
    runing_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward
        y_hat, _ = model(x)  # y_hat -> probabillity
        loss = criterion(y_hat, y)
        runing_loss += loss.item() * x.size(0)

        # Backprop
        loss.backward()
        optimizer.step()

    epoch_loss = runing_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validata(valid_loader, model, criterion, device):
    # Evaluation
    model.eval()
    runing_loss = 0

    for x, y_true in valid_loader:
        x = x.to(device)
        y = y_true.to(device)

        y_hat, _ = model(x)
        loss = criterion(y_hat, y)
        runing_loss += loss.item() * x.size(0)

    epoch_loss = runing_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    epochs,
    device,
    pring_every=1,
):
    train_losses = []
    valid_losses = []

    # Training
    for epoch in range(0, epochs):
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device
        )
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validata(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % pring_every == (pring_every - 1):
            train_acc = acc(model, train_loader, device=device)
            valid_acc = acc(model, valid_loader, device=device)

            print(
                f"{datetime.now().time().replace(microsecond=0)} ----------"
                f"Epoch : {epoch + 1}\t"
                f"Train loss : {train_loss:.4f}\t"
                f"Valid loss : {valid_loss:.4f}\t"
                f"Train ACC : {100 * train_acc:.2f}\t"
                f"Valid ACC : {100 * valid_acc:.2f}"
            )

    return model, optimizer, (train_losses, valid_losses)


torch.manual_seed(random_seed)
model = LeNet5(class_number).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
model, optimizer, _ = training_loop(
    model, criterion, optimizer, train_loader, valid_loader, class_number, device
)
