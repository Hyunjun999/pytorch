import torch
import albumentations as A
import torch.nn as nn
import torch.optim as optim
from torchvision.models.mobilenetv2 import mobilenet_v2
from torch.utils.data import DataLoader
from food_customdataset import CustomDataset
from tqdm import tqdm


def train(model, train_loader, val_loader, epoch, device, opt, criterion):
    best_val_acc = 0.0
    train_loss_l, val_loss_l = [], []
    train_acc_l, val_acc_l = [], []
    print("Train..")
    for e in range(epoch):
        train_loss, val_loss = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0

        model.train()

        # tqdml
        train_loader_iter = tqdm(
            train_loader, desc=f"Epoch : {e} / {epoch}", leave=False
        )
        for i, (data, target) in enumerate(train_loader_iter):
            data = data.float().to(device)
            target = target.float().to(device)

            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            opt.step()

            train_loss += loss.item()

            # Train_acc
            _, pred = torch.max(output, 1)
            train_acc += (pred == target).sum().item()
            train_loader_iter.set_postfix({"Loss": loss.item()})

            if i % 10 == 9:
                print(f"Epoch = {e + 1} / {epoch}, Loss = {loss.item()}")

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        # Eval
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data = data.float().to(device)
                target = target.float().to(device)

                outputs = model(data)
                pred = outputs.argmax(dim=1, keepdim=True)
                pred = torch.max(outputs, 1)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
                val_loss += criterion(outputs, target).item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        train_loss_l.append(train_loss)
        train_acc_l.append(train_acc)
        val_loss_l.append(val_loss)
        val_acc_l.append(val_acc)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "food_best_mobilenetv2.pt")
            best_val_acc = val_acc

        print(
            f"Epoch = {e + 1} / {epoch}, Train loss = {train_loss:.4f}, Train acc = {train_acc:.4f}, Val loss = {val_loss:.4f}, Val acc = {val_acc:.4f}"
        )
    print(best_val_acc)
    return model, train_loss_l, val_loss_l, train_acc_l, val_acc_l


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2(pretrained=True)
    num_features = 20
    model.classifier[1] = nn.Linear(1280, num_features)
    model.to(device)

    # aug w/ albumentations
    train_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=220),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6
            ),
            A.RandomShadow(),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Resize(height=224, width=224),
            # ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=220),
            A.Resize(height=224, width=224),
            #  ToTensorV2()
        ]
    )

    # Datasest && Dataloader
    train_dataset = CustomDataset("./food_dataset/train/", transform=train_transform)
    val_dataset = CustomDataset("./food_dataset/validation/", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=124, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=124, shuffle=False)

    # Hyperparams
    epoch = 20
    criterion = nn.CrossEntropyLoss().to(device)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

    train(model, train_loader, val_loader, epoch, device, opt, criterion)


if __name__ == "__main__":
    main()
