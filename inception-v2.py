import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(
        self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool
    ):
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.conv1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.conv3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)

        self.conv3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        # 5x5 convolution branch
        self.conv5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.conv5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)

        # Max Pooling branch
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, out_pool, kernel_size=1)

    def forward(self, x):
        # 1x1 convolution branch
        out_1x1 = F.relu(self.conv1x1(x))

        # 3x3 convolution branch
        out_3x3 = F.relu(self.conv3x3(F.relu(self.conv3x3_reduce(x))))

        # 5x5 convolution branch
        out_5x5 = F.relu(self.conv5x5(F.relu(self.conv5x5_reduce(x))))

        # Max Pooling branch
        out_pool = F.relu(self.conv_pool(self.pool(x)))

        # Concat branches
        out = torch.cat([out_1x1, out_3x3, out_5x5, out_pool], dim=1)

        return out


class IncetionV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(IncetionV2, self).__init__()

        #  Stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Inception modules

        # Inception modules(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool)
        self.inception3a = InceptionModule(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # avg pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Stem layers
        x = self.stem(x)

        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception5a(x)
        x = self.inception5b(x)

        # avg pool
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
