import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg11, resnet18
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CIFAR10 preprocessing
transform = transform.Compose(
    [transform.ToTensor(), transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = CIFAR10(
    root="./CIFAR10/", train=True, download=True, transform=transform
)
test_dataset = CIFAR10(
    root="./CIFAR10/", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# VGG11 && ResNet18 definition
vgg = vgg11(pretrained=False)
resent = resnet18(pretrained=False)
num_feautres_vgg = vgg.classifier[6].in_features
num_feautres_resnet = resent.fc.in_features
vgg.classifier[6] = nn.Linear(num_feautres_vgg, 10)  # CIFAR10 has 10 classes
resent.fc = nn.Linear(num_feautres_resnet, 10)  # CIFAR10 has 10 classes


# Voting Ensemble
class Ensemble(nn.Module):
    def __init__(self, models) -> None:
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        avg_output = torch.mean(outputs, dim=0)
        return avg_output


ensemble_model = Ensemble([vgg, resent])
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(ensemble_model.parameters(), lr=0.001)


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
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


def combine_pred(predictions):
    combined = torch.cat(predictions, dim=0)
    _, predicted_labels = torch.max(combined, 1)
    return predicted_labels


if __name__ == "__main__":
    for e in range(1, 2):
        print(f"Training model {e}")
        ensemble_model = ensemble_model.to(device)
        train(ensemble_model, device, train_loader, opt, criterion)
        predictions = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = ensemble_model(data)
                predictions.append(output)

        combine_pred = combine_pred(predictions)
        acc = accuracy_score(test_dataset.targets, combine_pred.cpu().numpy())
        print(f"Model {e} Acc: {acc}")
