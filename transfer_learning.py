import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)  # Define pre-trained model

# Freeze model params before get into the new task
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer for the # of classes in your new task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load and preprocess the data
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.fc.parameters(), lr=0.001)
model = model.to(device)
print(train_loader.dataset)
print(model)
# Training
epochs = 10
for e in range(epochs):
    train_loss = 0.0
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch = {e + 1} / {epochs}, Loss : {train_loss}")

# Evaluation
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
acc = 100 * correct / total
print(f"Test acc : {acc}")
