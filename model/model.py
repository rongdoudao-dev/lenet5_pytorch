import torch
import torch.nn as nn
from model.backbone import BackBone
from model.head import Head


class LeNet5(nn.Module):
    """
    LeNet-5 完整网络
    BackBone (卷积) + Head (全连接)
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # 卷积特征提取部分
        self.backbone = BackBone()

        # 全连接分类部分
        self.head = Head(num_classes=num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入图像 [batch_size, 1, 32, 32]
        :return: logits [batch_size, num_classes]
        """
        # 卷积提取特征
        features = self.backbone(x)

        # 全连接分类
        logits = self.head(features)

        return logits
