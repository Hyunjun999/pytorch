import torch
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import vgg11
from audio_customdataset import CustomDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model setting
    model = vgg11()
    print(model)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 3)

    # .pt load
    model.load_state_dict(torch.load(f="./audio_best.pt"))

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    test_dataset = CustomDataset("./audio_dataset/val", val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            _, pred = torch.max(output, 1)

            correct += (pred == target).sum().item()

    print(f"Test acc : {correct / len(test_loader.dataset)}")


if __name__ == "__main__":
    main()
