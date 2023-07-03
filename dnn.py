import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader


class RBM(nn.Module):
    def __init__(self, visible_size, hidden_size):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_size, hidden_size))
        self.v_bias = nn.Parameter(torch.randn(visible_size))
        self.h_bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        hidden_prob = torch.sigmoid(torch.matmul(x, self.W) + self.h_bias)
        hidden_state = torch.bernoulli(hidden_prob)
        visible_prob = torch.sigmoid(
            torch.matmul(hidden_state, torch.transpose(self.W, 0, 1)) + self.v_bias
        )
        return visible_prob, hidden_state


if __name__ == "__main__":
    # Load the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data/", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # RBM instance
    visible_size = 28 * 28
    hidden_size = 256
    rbm = RBM(visible_size, hidden_size)

    criterion = nn.BCELoss()
    opt = torch.optim.SGD(
        rbm.parameters(),
        lr=0.01,
    )

    # Train
    epochs = 10
    for e in range(epochs):
        for img, _ in train_loader:
            inputs = img.view(-1, visible_size)

            # Forward
            visible_prob, _ = rbm(inputs)
            # Backprop
            loss = criterion(visible_prob, inputs)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch = {e + 1} / {epochs}, Loss = {loss.item()}")
        vutils.save_image(
            rbm.W.view(hidden_size, 1, 28, 28),
            f"weight_epoch_{e + 1}.png",
            normarlize=True,
        )
        # Comprise 28 * 28 tensor
        inputs_display = inputs.view(-1, 1, 28, 28)
        outputs_display = visible_prob.view(-1, 1, 28, 28)
        comparsion = torch.cat([inputs_display, outputs_display], dim=3)
        vutils.save_image(
            comparsion,
            f"reconstruction_epoch_{e + 1}.png",
            normarlize=True,
        )
