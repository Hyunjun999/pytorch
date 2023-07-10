import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device >>", device)

train_transform = transform.Compose(
    [
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.RandAugment(),
        transform.ToTensor(),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_transform = transform.Compose(
    [transform.ToTensor(), transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = CIFAR10(
    root="./data", train=True, download=False, transform=train_transform
)
test_dataset = CIFAR10(
    root="./data", train=False, download=False, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

model = resnet18(pretrained=True)  # Pre-trained Resnet with 18 layers model
num_features = model.fc.in_features
model.fc = nn.Linear(
    num_features, 10
)  # Connect fully connected layer to make output as 10 for the CIFAR10 dataset
# print("fc in features >>", num_features)
# print(model)

bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=7), n_estimators=5
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, device, test_loader):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

    acc = accuracy_score(targets, predictions)
    return acc


def ensemble_pred(models, device, test_loader):
    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = []
            for model in models:
                model = model.to(device)
                model.eval()
                output = model(data)
                outputs.append(output)

            ensemble_output = torch.stack(outputs).mean(dim=0)
            _, pred = torch.max(ensemble_output, 1)
            predictions.extend(pred.cpu().numpy())


if __name__ == "__main__":
    models = []
    for epoch in range(1, 20):
        print(f"Train ... {epoch}")
        model = model.to(device)
        train(model, device, train_loader, optimizer, criterion)
        acc = evaluate(model, device, test_loader)
        print(f"Model {epoch} ACC {acc:.2f}")
        models.append(model)

    ensemble_predictions = ensemble_pred((models, device, test_loader))
    ensemble_acc = accuracy_score(test_dataset.targets, ensemble_predictions)
    print(f"\nEnsemble Acc : {ensemble_acc:.2f}")
