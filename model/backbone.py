import torch
import torch.nn as nn


class BackBone(nn.Module):
    """
    LeNet-5 卷积特征提取骨干网络
    输入: [batch_size, 1, 32, 32]  (灰度图 32x32)
    输出: [batch_size, 16, 5, 5]    (16通道 5x5 特征图)
    """

    def __init__(self):
        super(BackBone, self).__init__()

        # C1: 卷积层 1
        # 输入: 1通道, 输出: 6通道, 卷积核 5x5, 步长 1, padding 0
        # 输出尺寸: (32 - 5) / 1 + 1 = 28
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )

        # S2: 平均池化层
        # 卷积核 2x2, 步长 2
        # 输出尺寸: 28 / 2 = 14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: 卷积层 2
        # 输入: 6通道, 输出: 16通道, 卷积核 5x5, 步长 1, padding 0
        # 输出尺寸: (14 - 5) / 1 + 1 = 10
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )

        # S4: 平均池化层
        # 卷积核 2x2, 步长 2
        # 输出尺寸: 10 / 2 = 5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 激活函数: Sigmoid (LeNet-5 原版用 Sigmoid)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        :param x: 输入图像 [batch_size, 1, 32, 32]
        :return: 特征图 [batch_size, 16, 5, 5]
        """
        # C1 + Sigmoid + S2
        x = self.conv1(x)
        x = self.activate(x)
        x = self.pool1(x)

        # C3 + Sigmoid + S4
        x = self.conv2(x)
        x = self.activate(x)
        x = self.pool2(x)

        return x
