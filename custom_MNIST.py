import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import os


class HandWriteDataSet(Dataset):
    def __init__(self, path, transform=None) -> None:
        self.path = path
        self.dir_list = os.listdir(path)
        self.transform = transform

    def __getitem__(self, index):
        filename = self.dir_list[index]
        # label = list(
        #     map(
        #         lambda x: int(x[0]),
        #         map(
        #             lambda x: x.split("."),
        #             map(
        #                 lambda x: x[-1],
        #                 map(
        #                     lambda x: x.split("_"),
        #                     map(
        #                         lambda x: x[-1],
        #                         map(
        #                             lambda x: x.split("/"),
        #                             glob.glob(
        #                                 os.path.join("./handwriting_data/", "*_*.*")
        #                             ),
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )
        # )[index]
        if len(filename.split("_")) > 1:
            label = int(filename.split("_")[-1].split(".")[0])
        else:
            label = int(filename.split(".")[0][-1])
        img = Image.open((os.path.join("./handwriting_data/", filename))).convert("L")
        img = img.resize((28, 28))

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dir_list)


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = CNN().to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    total_dataset = HandWriteDataSet("./handwriting_data/", transform=transform)

    test_len = int(len(total_dataset) * 0.2)
    train_len = len(total_dataset) - test_len

    train_subset, test_subset = random_split(total_dataset, [train_len, test_len])

    train_dataset = train_subset.dataset
    test_dataset = test_subset.dataset

    # temp_dict = dict.fromkeys(list(range(10)), 0)
    # for _, label in test_dataset:
    #     temp_dict[label] += 1

    # print(temp_dict)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        print(f"Epoch {epoch+1} / {10}, Loss : {epoch_loss:.4f}")

    # Model Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f} %")
