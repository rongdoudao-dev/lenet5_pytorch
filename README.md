# LeNet-5 PyTorch 实现

## 项目结构

```
lenet5_pytorch/
├── model/
│   ├── __init__.py
│   ├── backbone.py      # 卷积特征提取模块
│   ├── head.py          # 全连接分类头模块
│   └── model.py         # 完整 LeNet-5 模型
├── utils/
│   └── __init__.py
├── runs/                # 训练输出目录
│   ├── best.pt          # 最佳模型权重
│   ├── last.pt          # 最后一轮模型权重
│   └── training_curves.png  # 训练曲线图
├── train.py             # 训练主程序
└── README.md            # 说明文档
```

## 网络结构

| 层名 | 类型 | 输入尺寸 | 输出尺寸 | 核大小 | 步长 | 填充 |
|---|---|---|---|---|---|---|
| C1 | 卷积 | 1×32×32 | 6×28×28 | 5×5 | 1 | 0 |
| S2 | 平均池化 | 6×28×28 | 6×14×14 | 2×2 | 2 | 0 |
| C3 | 卷积 | 6×14×14 | 16×10×10 | 5×5 | 1 | 0 |
| S4 | 平均池化 | 16×10×10 | 16×5×5 | 2×2 | 2 | 0 |
| C5 | 全连接 | 400 | 120 | - | - | - |
| F6 | 全连接 | 120 | 84 | - | - | - |
| Output | 全连接 | 84 | 10 | - | - | - |

## 运行方式

```bash
# 安装依赖
pip install torch torchvision matplotlib

# 训练
python train.py
```

## 训练结果

- 数据集: MNIST
- Epochs: 10
- Batch Size: 64
- Optimizer: Adam
- Learning Rate: 0.001
