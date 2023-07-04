import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        # Original img size = 28 * 28
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # Input channel = 1 -> black & white img, Input img size will be (28 + 2 - 3 + 1) = 28 * 28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Img size will be (28 / 2) * (28 / 2) = 14 * 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Img size will be 14 + 2 - 3 + 1 = 14 * 14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Img size will be (14 / 2) * (14 / 2) = 7 * 7
        self.fc = nn.Linear(7 * 7 * 32, 10)
        # From the above, we have 7 * 7 img and the output channel is still 32, so flatten as 7 * 7 * 32
        self.conv1_output = None
        self.conv2_output = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        self.conv1_output = x

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        self.conv2_output = x

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    train_dataset = MNIST(
        "./MNIST_train", train=True, transform=ToTensor(), download=True
    )
    test_dataset = MNIST(
        "./MNIST_test", train=False, transform=ToTensor(), download=True
    )

    # Hyperparams
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    fig.tight_layout(pad=4.0)
    axs = axs.flatten()
    epoch_losses = []

    for e in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            opt.zero_grad()
            # Get the predicted values as 'outputs'
            outputs = model(img)
            # Compaing above one with the actual label
            loss = criterion(outputs, label)
            loss.backward()
            opt.step()
            running_loss += loss.item() * img.size(0)
        epoch_loss = running_loss / len(train_dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch =  {e + 1} / {num_epochs}, Loss = {epoch_loss}")

        if e == 0:
            # Conv1 weight visualization
            weights = model.conv1.weight.detach().cpu().numpy()
            axs[0].imshow(weights[0, 0], cmap="coolwarm")
            axs[0].set_title("Conv1 Weights")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(axs[0].imshow(weights[0, 0], cmap="coolwarm"), cax=cax)

            # Conv2 weight visualization
            weights = model.conv2.weight.detach().cpu().numpy()
            axs[1].imshow(weights[0, 0], cmap="coolwarm")
            axs[1].set_title("Conv2 Weights")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(axs[1].imshow(weights[0, 0], cmap="coolwarm"), cax=cax)

            # Conv1 output
            if model.conv1_output is not None:
                conv1_output = model.conv1_output.detach().cpu().numpy()
                axs[2].imshow(conv1_output[0, 0], cmap="coolwarm")
                axs[2].set_title("Conv1 output")
                divider = make_axes_locatable(axs[2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(
                    axs[2].imshow(conv1_output[0, 0], cmap="coolwarm"), cax=cax
                )

            # Conv2 output
            if model.conv2_output is not None:
                conv2_output = model.conv2_output.detach().cpu().numpy()
                axs[3].imshow(conv2_output[0, 0], cmap="coolwarm")
                axs[3].set_title("Conv2 output")
                divider = make_axes_locatable(axs[3])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(
                    axs[3].imshow(conv2_output[0, 0], cmap="coolwarm"), cax=cax
                )
        axs[4].plot(range(e + 1), epoch_losses)
        axs[4].set_title("training_loss")
        axs[4].set_xlabel("Epoch")
        axs[4].set_ylabel("Loss")

    plt.show()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    acc = 100.0 * correct / total
    print(f"Acc : {acc:.2f}%")
