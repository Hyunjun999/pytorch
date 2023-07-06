import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1) -> None:
        super(BasicBlock, self).__init__()
        # Input channel = # of channels at inout feature map
        # out channel = # of channels at output feature map
        # bias TRUE -> there is a bias to train model more flexible
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(
            out_channel
        )  # Batch normalization can resolve gradient vanishing, and
        # it calculates mean and var for each channel
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()  # Residual connection

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        # F(x) = conv1 -> relu -> conv2, H(x) = conv1 -> relu -> conv2 -> relu, H(x) = F(x) + x
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block: BasicBlock, layer: list, num_class=1000) -> None:
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(
            3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layer[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layer[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, out_channels, stride))
        self.in_channel = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channel, out_channels))
        print(layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet34(num_class=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class)


model = resnet34()
intputs = torch.randn(1, 3, 224, 224)
output = model(intputs)
print(output.shape)
