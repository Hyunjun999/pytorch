import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet50, mobilenet_v2, efficientnet_b1
from torch.utils.data import DataLoader
from customset import Fashionset
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_loss(train_loss_l, val_loss_l, model, batch):
    plt.figure()
    plt.plot(train_loss_l, label="Train loss")
    plt.plot(val_loss_l, label="Val loss")
    plt.plot(
        max(train_loss_l),
        marker="o",
        markersize=8,
        markeredgecolor="black",
        markerfacecolor="red",
    )
    plt.plot(
        max(val_loss_l),
        marker="o",
        markersize=8,
        markeredgecolor="red",
        markerfacecolor="green",
    )
    plt.title(f"{model.__class__.__name__}_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./{model.__class__.__name__}_loss_w_{batch}.jpg")


def plot_acc(train_acc_l, val_acc_l, model, batch):
    plt.figure()
    plt.plot(train_acc_l, label="Train acc")
    plt.plot(val_acc_l, label="Val acc")
    plt.plot(
        max(train_acc_l),
        marker="o",
        markersize=8,
        markeredgecolor="black",
        markerfacecolor="red",
    )
    plt.plot(
        max(val_acc_l),
        marker="o",
        markersize=8,
        markeredgecolor="red",
        markerfacecolor="green",
    )

    plt.title(f"{model.__class__.__name__}_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.savefig(f"./{model.__class__.__name__}_acc_w_{batch}.jpg")


def train(model, train_loader, val_loader, epochs, criterion, batch):
    model.to(device)
    print(model)
    best_val_acc = 0.0
    train_acc_l, train_loss_l, val_acc_l, val_loss_l = [], [], [], []
    opt = optim.Adam(model.parameters(), lr=0.001)
    # loop over the dataset multiple times
    for e in range(epochs):
        train_acc, val_acc, train_loss, val_loss = [0.0] * 4
        model.train()
        for i, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.float().to(device), target.to(device)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_acc += (pred == target).sum().item()
            print(f"Epoch = {e + 1} / {epochs}, Loss = {loss.item()}")

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        # Eval
        model.eval()
        with torch.no_grad():
            for data, label in tqdm(val_loader):
                data, label = data.float().to(device), label.to(device)

                outputs = model(data)
                pred = outputs.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(label.view_as(pred)).sum().item()
                val_loss += criterion(outputs, label).item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        train_loss_l.append(train_loss)
        train_acc_l.append(train_acc)
        val_loss_l.append(val_loss)
        val_acc_l.append(val_acc)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), f"./{model.__class__.__name__}.pt")
            best_val_acc = val_acc

        print(
            f"Epoch = {e + 1} / {epochs}, Train loss = {train_loss:.4f}, Train acc = {train_acc:.4f}, Val loss = {val_loss:.4f}, Val acc = {val_acc:.4f}"
        )
        plot_acc(train_acc_l, val_acc_l, model, batch)
        plot_loss(train_loss_l, val_loss_l, model, batch)
    print("Highest_val_acc: ", best_val_acc)


if __name__ == "__main__":
    # Hyperparams
    train_transform = A.Compose(
        [
            A.Normalize(),
            A.Resize(640, 640),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(p=0.15),
            A.Rotate(limit=90, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose([A.Normalize(), A.Resize(640, 640), ToTensorV2()])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    # r = resnet50(pretrained=True)
    # r.fc = nn.Linear(2048, 6)
    m = mobilenet_v2(pretrained=True)
    m.classifier[1] = nn.Linear(1280, 21)
    # e = efficientnet_b1(pretrained=True)
    # e.classifier[1] = nn.Linear(1280, 6)
    model = [m]
    epochs = 200
    batch = 16
    # Dataset && Dataloader
    train_dataset = Fashionset("./cloth_category/train/", transform=train_transform)
    val_dataset = Fashionset("./cloth_category/val/", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    for m in model:
        print(m.__class__.__name__)
        train(m, train_loader, val_loader, epochs, criterion, batch)
