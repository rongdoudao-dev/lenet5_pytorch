"""
LeNet-5 特征图可视化脚本
功能:
1. 可视化每一层的特征图 (C1/S2/C3/S4)
2. 6通道取 mean/max 合成一张热力图
3. 完整数据流: 原图 -> 信号 -> C1(6张) -> S2(6张) -> C3(16张) -> S4(16张)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model.model import LeNet5
import os


def get_device():
    """获取设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path="runs/best.pt", device="cpu"):
    """加载训练好的模型"""
    model = LeNet5(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_test_image(index=0, device="cpu"):
    """获取一张测试图片"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    image, label = test_set[index]
    return image.unsqueeze(0).to(device), label


def extract_features(model, x):
    """提取每一层的特征图"""
    features = {}

    # 原图 (反归一化)
    original = x.cpu().squeeze().numpy() * 0.3081 + 0.1307
    original = np.clip(original, 0, 1)
    features['original'] = original

    # 信号 (归一化后)
    features['signal'] = x.cpu().squeeze().numpy()

    # BackBone 各层
    with torch.no_grad():
        # C1: 卷积1
        x_c1 = model.backbone.conv1(x)
        features['C1'] = x_c1.cpu().squeeze().numpy()  # (6, 28, 28)

        # ReLU1
        x_relu1 = model.backbone.relu1(x_c1)

        # S2: 池化1
        x_s2 = model.backbone.pool1(x_relu1)
        features['S2'] = x_s2.cpu().squeeze().numpy()  # (6, 14, 14)

        # C3: 卷积2
        x_c3 = model.backbone.conv2(x_s2)
        features['C3'] = x_c3.cpu().squeeze().numpy()  # (16, 10, 10)

        # ReLU2
        x_relu2 = model.backbone.relu2(x_c3)

        # S4: 池化2
        x_s4 = model.backbone.pool2(x_relu2)
        features['S4'] = x_s4.cpu().squeeze().numpy()  # (16, 5, 5)

    return features


def plot_feature_maps(features, save_dir="runs/features"):
    """绘制所有特征图"""
    os.makedirs(save_dir, exist_ok=True)

    # ===== 1. 完整数据流大图 (14+张) =====
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('LeNet-5 完整数据流: 图像信息 -> 信号信息 -> 局部特征 -> 语义信息', fontsize=14)

    # 第1行: 原图 + 信号 (2张)
    ax = fig.add_subplot(5, 8, 1)
    ax.imshow(features['original'], cmap='gray')
    ax.set_title('① 原图 32x32\n(只有0和255)', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(5, 8, 2)
    ax.imshow(features['signal'], cmap='gray')
    ax.set_title('② 信号 -1~1\n(归一化后)', fontsize=9)
    ax.axis('off')

    # 第2行: C1 6张 (28x28)
    c1 = features['C1']
    for i in range(6):
        ax = fig.add_subplot(5, 8, 9 + i)
        ax.imshow(c1[i], cmap='viridis')
        ax.set_title(f'③ C1-{i+1}\n28x28', fontsize=8)
        ax.axis('off')

    # 第3行: S2 6张 (14x14)
    s2 = features['S2']
    for i in range(6):
        ax = fig.add_subplot(5, 8, 17 + i)
        ax.imshow(s2[i], cmap='viridis')
        ax.set_title(f'④ S2-{i+1}\n14x14', fontsize=8)
        ax.axis('off')

    # 第4行: C3 前8张 (10x10)
    c3 = features['C3']
    for i in range(8):
        ax = fig.add_subplot(5, 8, 25 + i)
        ax.imshow(c3[i], cmap='viridis')
        ax.set_title(f'⑤ C3-{i+1}\n10x10', fontsize=8)
        ax.axis('off')

    # 第5行: C3 后8张 (10x10)
    for i in range(8):
        ax = fig.add_subplot(5, 8, 33 + i)
        ax.imshow(c3[i + 8], cmap='viridis')
        ax.set_title(f'⑤ C3-{i+9}\n10x10', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/full_dataflow.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 完整数据流图已保存: {save_dir}/full_dataflow.png")

    # ===== 2. C1层6张单独大图 =====
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('C1层特征图 (6个卷积核 -> 6张28x28)', fontsize=14)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(c1[i], cmap='viridis')
        ax.set_title(f'C1-{i+1} (28x28)', fontsize=11)
        ax.axis('off')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/C1_feature_maps.png", dpi=150)
    plt.close()
    print(f"✓ C1特征图已保存: {save_dir}/C1_feature_maps.png")

    # ===== 3. S2层6张单独大图 =====
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('S2层特征图 (6个通道 -> 6张14x14)', fontsize=14)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(s2[i], cmap='viridis')
        ax.set_title(f'S2-{i+1} (14x14)', fontsize=11)
        ax.axis('off')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/S2_feature_maps.png", dpi=150)
    plt.close()
    print(f"✓ S2特征图已保存: {save_dir}/S2_feature_maps.png")

    # ===== 4. C3层16张大图 =====
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('C3层特征图 (16个卷积核 -> 16张10x10)', fontsize=14)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(c3[i], cmap='viridis')
        ax.set_title(f'C3-{i+1} (10x10)', fontsize=10)
        ax.axis('off')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/C3_feature_maps.png", dpi=150)
    plt.close()
    print(f"✓ C3特征图已保存: {save_dir}/C3_feature_maps.png")

    # ===== 5. S4层16张大图 =====
    s4 = features['S4']
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('S4层特征图 (16个通道 -> 16张5x5)', fontsize=14)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(s4[i], cmap='viridis')
        ax.set_title(f'S4-{i+1} (5x5)', fontsize=10)
        ax.axis('off')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/S4_feature_maps.png", dpi=150)
    plt.close()
    print(f"✓ S4特征图已保存: {save_dir}/S4_feature_maps.png")

    # ===== 6. 6通道 Mean/Max 合成热力图 =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('6通道特征图融合: Mean vs Max', fontsize=14)

    # C1 Mean
    c1_mean = np.mean(c1, axis=0)
    ax = axes[0, 0]
    im = ax.imshow(c1_mean, cmap='hot')
    ax.set_title('C1 Mean (6通道平均)\n28x28', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # C1 Max
    c1_max = np.max(c1, axis=0)
    ax = axes[0, 1]
    im = ax.imshow(c1_max, cmap='hot')
    ax.set_title('C1 Max (6通道最大)\n28x28', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # C1 原图对比
    ax = axes[0, 2]
    ax.imshow(features['original'], cmap='gray')
    ax.set_title('原图对比\n32x32', fontsize=11)
    ax.axis('off')

    # S2 Mean
    s2_mean = np.mean(s2, axis=0)
    ax = axes[1, 0]
    im = ax.imshow(s2_mean, cmap='hot')
    ax.set_title('S2 Mean (6通道平均)\n14x14', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # S2 Max
    s2_max = np.max(s2, axis=0)
    ax = axes[1, 1]
    im = ax.imshow(s2_max, cmap='hot')
    ax.set_title('S2 Max (6通道最大)\n14x14', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # S2 原图对比
    ax = axes[1, 2]
    ax.imshow(features['original'], cmap='gray')
    ax.set_title('原图对比\n32x32', fontsize=11)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/channel_fusion_mean_max.png", dpi=150)
    plt.close()
    print(f"✓ 6通道融合图已保存: {save_dir}/channel_fusion_mean_max.png")

    # ===== 7. 16通道 Mean/Max 合成热力图 =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('16通道特征图融合: Mean vs Max', fontsize=14)

    # C3 Mean
    c3_mean = np.mean(c3, axis=0)
    ax = axes[0, 0]
    im = ax.imshow(c3_mean, cmap='hot')
    ax.set_title('C3 Mean (16通道平均)\n10x10', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # C3 Max
    c3_max = np.max(c3, axis=0)
    ax = axes[0, 1]
    im = ax.imshow(c3_max, cmap='hot')
    ax.set_title('C3 Max (16通道最大)\n10x10', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # C3 原图对比
    ax = axes[0, 2]
    ax.imshow(features['original'], cmap='gray')
    ax.set_title('原图对比\n32x32', fontsize=11)
    ax.axis('off')

    # S4 Mean
    s4_mean = np.mean(s4, axis=0)
    ax = axes[1, 0]
    im = ax.imshow(s4_mean, cmap='hot')
    ax.set_title('S4 Mean (16通道平均)\n5x5', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # S4 Max
    s4_max = np.max(s4, axis=0)
    ax = axes[1, 1]
    im = ax.imshow(s4_max, cmap='hot')
    ax.set_title('S4 Max (16通道最大)\n5x5', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # S4 原图对比
    ax = axes[1, 2]
    ax.imshow(features['original'], cmap='gray')
    ax.set_title('原图对比\n32x32', fontsize=11)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/channel_fusion_16_mean_max.png", dpi=150)
    plt.close()
    print(f"✓ 16通道融合图已保存: {save_dir}/channel_fusion_16_mean_max.png")

    print(f"\n{'='*50}")
    print(f"所有特征图已保存到: {save_dir}/")
    print(f"{'='*50}")


def main():
    device = get_device()
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model("runs/best.pt", device)
    print("✓ 模型加载成功")

    # 获取测试图片 (取第0张, 可以改成任意数字)
    image, label = get_test_image(index=0, device=device)
    print(f"✓ 测试图片加载成功, 真实标签: {label}")

    # 提取特征
    features = extract_features(model, image)
    print("✓ 特征提取完成")

    # 绘制所有特征图
    plot_feature_maps(features)


if __name__ == "__main__":
    main()
