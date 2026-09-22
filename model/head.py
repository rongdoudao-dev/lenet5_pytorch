import torch
import torch.nn as nn


class Head(nn.Module):
    """
    LeNet-5 全连接分类头
    输入: [batch_size, 16, 5, 5]  (BackBone输出特征图)
    输出: [batch_size, 10]         (10分类 logits)
    """

    def __init__(self, num_classes=10):
        super(Head, self).__init__()

        # 展开维度: 16 * 5 * 5 = 400
        self.flatten = nn.Flatten()

        # C5: 全连接层 1
        # 输入: 400, 输出: 120
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        # F6: 全连接层 2
        # 输入: 120, 输出: 84
        self.fc2 = nn.Linear(in_features=120, out_features=84)

        # Output: 输出层
        # 输入: 84, 输出: num_classes (10)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

        # 激活函数: ReLU (比 Sigmoid 更快更好, 准确率更高)
        self.activate = nn.ReLU()

    def forward(self, x):
        """
        前向传播
        :param x: 特征图 [batch_size, 16, 5, 5]
        :return: logits [batch_size, num_classes]
        """
        # 展平: [batch, 16, 5, 5] -> [batch, 400]
        x = self.flatten(x)

        # C5 + ReLU
        x = self.fc1(x)
        x = self.activate(x)

        # F6 + ReLU
        x = self.fc2(x)
        x = self.activate(x)

        # Output (不加激活, CrossEntropyLoss 内部自带 Softmax)
        x = self.fc3(x)

        return x
