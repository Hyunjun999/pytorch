import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datatsets
import torchvision.transforms as transforms


class DropoutNet(nn.Module):
    def __init__(self) -> None:
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NonDropoutNet(nn.Module):
    def __init__(self) -> None:
        super(NonDropoutNet, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


train_transform = transforms.Compose(
    [
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.3,)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.3,)),
    ]
)

train_dataset = datatsets.MNIST(
    root="./dropout_MNIST", train=True, download=False, transform=train_transform
)
test_dataset = datatsets.MNIST(
    root="./dropout_MNIST", train=False, download=False, transform=test_transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Dropout
dropout_model = DropoutNet()
print(dropout_model)

# Hyperparams
dropout_criterion = nn.CrossEntropyLoss()
dropout_optimizer = optim.SGD(dropout_model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    dropout_model.train()
    for img, label in train_loader:
        dropout_optimizer.zero_grad()
        output = dropout_model(img)
        loss = dropout_criterion(output, label)
        loss.backward()
        dropout_optimizer.step()

# Evaluation
dropout_model.eval()
with torch.no_grad():
    correct, total = 0, 0
    for img, label in test_loader:
        output = dropout_model(img)
        _, prediction = torch.max(output.data, 1)
        total += img.size(0)
        correct += (prediction == label).sum().item()
    print("DropoutNet Acc :", 100 * correct / total)

# Non-dropout
non_dropout_model = NonDropoutNet()
print(non_dropout_model)

# Hyperparams
non_dropout_criterion = nn.CrossEntropyLoss()
non_dropout_optimizer = optim.SGD(non_dropout_model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    non_dropout_model.train()
    for img, label in train_loader:
        non_dropout_optimizer.zero_grad()
        output = non_dropout_model(img)
        loss = non_dropout_criterion(output, label)
        loss.backward()
        non_dropout_optimizer.step()

# Evaluation
non_dropout_model.eval()
with torch.no_grad():
    correct, total = 0, 0
    for img, label in test_loader:
        output = non_dropout_model(img)
        _, prediction = torch.max(output.data, 1)
        total += img.size(0)
        correct += (prediction == label).sum().item()
    print("NonDropoutNet Acc :", 100 * correct / total)
