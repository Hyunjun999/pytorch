import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 data and apply transform to them respectively
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ]
)
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))]
)
train = torchvision.datasets.CIFAR10(
    root="./CIFAR10", train=True, download=True, transform=train_transform
)
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

test = torchvision.datasets.CIFAR10(
    root="./CIFAR10", train=False, download=True, transform=test_transform
)
test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)


class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_eval(model: LeNet):
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(5):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            img, label = data
            opt.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if i % 200 == 190:
                print("[%d, %5d] loss : %.3f" % (e + 1, i, running_loss / 200))
                running_loss = 0.0
    print("Training done...")
    print("Evaluation start...")
    # Eval
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    acc = 100 * correct / total
    print("Acc :", acc)


model = LeNet()
train_and_eval(model)
