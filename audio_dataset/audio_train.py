import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import vgg11, VGG11_Weights
from audio_customdataset import CustomDataset


def train(model, train_loader, val_loader, epoch, device, opt, criterion):
    best_val_acc = 0.0
    train_loss_l, val_loss_l = [], []
    train_acc_l, val_acc_l = [], []
    print("Train..")
    for e in range(epoch):
        train_loss, val_loss = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0
        model.train()

        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            opt.step()

            train_loss += loss.item()

            # Train_acc
            _, pred = torch.max(output, 1)
            train_acc += (pred == target).sum().item()

            if i % 10 == 9:
                print(f"Epoch = {e + 1} / {epoch}, Loss = {loss.item()}")

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        # Eval
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                outputs = model(data)
                pred = outputs.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
                val_loss += criterion(outputs, target).item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        train_loss_l.append(train_loss)
        train_acc_l.append(train_acc)
        val_loss_l.append(val_loss)
        val_acc_l.append(val_acc)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "audio_best.pt")
            best_val_acc = val_acc

        print(
            f"Epoch = {e + 1} / {epoch}, Train loss = {train_loss:.4f}, Train acc = {train_acc:.4f}, Val loss = {val_loss:.4f}, Val acc = {val_acc:.4f}"
        )
    print(best_val_acc)
    return model, train_loss_l, val_loss_l, train_acc_l, val_acc_l


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")  # mac m1 or m2

    model = vgg11(weights=VGG11_Weights.DEFAULT)
    num_feature = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_feature, 3)
    model.to(device)

    # Transforms
    train_transform = transforms.Compose(
        [transforms.Resize((224, 244)), transforms.ToTensor()]
    )

    val_transform = transforms.Compose(
        [transforms.Resize((224, 244)), transforms.ToTensor()]
    )

    # Dataset
    train_dataset = CustomDataset("./audio_dataset/train", transform=train_transform)
    val_dataset = CustomDataset("./audio_dataset/val", transform=val_transform)

    print("Call dataset..")

    train_loader = DataLoader(
        train_dataset, batch_size=100, num_workers=4, pin_memory=True, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=100, num_workers=4, pin_memory=True, shuffle=False
    )
    print("Data loading done..")
    # Hyperparams
    criterion = nn.CrossEntropyLoss().to(device)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    epoch = 5

    train(model, train_loader, val_loader, epoch, device, opt, criterion)


if __name__ == "__main__":
    main()
