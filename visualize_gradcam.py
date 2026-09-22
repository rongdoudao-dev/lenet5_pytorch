"""
LeNet-5 Grad-CAM 梯度可视化脚本
功能:
1. Grad-CAM 类激活图 (显示模型关注图片的哪个区域)
2. 梯度可视化 (每层权重的梯度分布)
3. 国宝鉴定: 卷积梯度特征可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model.model import LeNet5
import os


class GradCAM:
    """Grad-CAM 类激活图"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """生成 Grad-CAM 热力图"""
        self.model.eval()

        # 前向传播
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重 (全局平均池化梯度)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 归一化到 [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 上采样到原图大小
        cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)

        return cam.squeeze().cpu().numpy(), target_class


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


def get_test_images(num_images=10, device="cpu"):
    """获取多张测试图片 (每个数字一张)"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    images = []
    labels = []
    found = set()

    for i in range(len(test_set)):
        img, label = test_set[i]
        if label not in found:
            images.append(img)
            labels.append(label)
            found.add(label)
        if len(found) == num_images:
            break

    return torch.stack(images).to(device), labels


def plot_gradcam_all_digits(model, device, save_dir="runs/gradcam"):
    """绘制所有数字的 Grad-CAM 热力图"""
    os.makedirs(save_dir, exist_ok=True)

    # 获取每个数字一张图
    images, labels = get_test_images(num_images=10, device=device)

    # 创建 Grad-CAM (目标层: conv2, 最后一层卷积)
    grad_cam = GradCAM(model, model.backbone.conv2)

    fig, axes = plt.subplots(3, 10, figsize=(25, 8))
    fig.suptitle('Grad-CAM 类激活图: 模型关注图片的哪个区域 (国宝鉴定)', fontsize=14)

    for i in range(10):
        input_img = images[i:i+1]
        cam, pred_class = grad_cam.generate(input_img, target_class=labels[i])

        # 原图 (反归一化)
        original = images[i].cpu().squeeze().numpy() * 0.3081 + 0.1307
        original = np.clip(original, 0, 1)

        # 第1行: 原图
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'数字 {labels[i]}\n(原图)', fontsize=10)
        axes[0, i].axis('off')

        # 第2行: Grad-CAM 热力图
        axes[1, i].imshow(cam, cmap='jet')
        axes[1, i].set_title(f'Grad-CAM\n(预测:{pred_class})', fontsize=10)
        axes[1, i].axis('off')

        # 第3行: 叠加图
        axes[2, i].imshow(original, cmap='gray')
        axes[2, i].imshow(cam, cmap='jet', alpha=0.5)
        axes[2, i].set_title('叠加图\n(关注区域)', fontsize=10)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/gradcam_all_digits.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Grad-CAM全数字图已保存: {save_dir}/gradcam_all_digits.png")


def plot_gradient_distribution(model, device, save_dir="runs/gradcam"):
    """绘制各层权重梯度分布"""
    os.makedirs(save_dir, exist_ok=True)

    # 获取一张测试图
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    image, label = test_set[0]
    image = image.unsqueeze(0).to(device)

    # 前向传播 + 反向传播
    model.train()
    output = model(image)
    loss = F.cross_entropy(output, torch.tensor([label]).to(device))
    loss.backward()

    # 收集各层梯度
    layer_names = []
    grad_means = []
    grad_stds = []
    grad_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_means.append(param.grad.mean().item())
            grad_stds.append(param.grad.std().item())
            grad_norms.append(param.grad.norm().item())

    # 绘制梯度分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('各层梯度分布 (梯度消失/爆炸检测)', fontsize=14)

    # 梯度均值
    axes[0].barh(range(len(layer_names)), grad_means, color='steelblue')
    axes[0].set_yticks(range(len(layer_names)))
    axes[0].set_yticklabels(layer_names, fontsize=8)
    axes[0].set_xlabel('Gradient Mean')
    axes[0].set_title('梯度均值')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # 梯度标准差
    axes[1].barh(range(len(layer_names)), grad_stds, color='coral')
    axes[1].set_yticks(range(len(layer_names)))
    axes[1].set_yticklabels(layer_names, fontsize=8)
    axes[1].set_xlabel('Gradient Std')
    axes[1].set_title('梯度标准差')

    # 梯度L2范数 (对数坐标)
    axes[2].barh(range(len(layer_names)), grad_norms, color='seagreen')
    axes[2].set_yticks(range(len(layer_names)))
    axes[2].set_yticklabels(layer_names, fontsize=8)
    axes[2].set_xlabel('Gradient L2 Norm (log scale)')
    axes[2].set_title('梯度L2范数 (对数坐标)')
    axes[2].set_xscale('log')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/gradient_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 梯度分布图已保存: {save_dir}/gradient_distribution.png")


def plot_conv_kernel_visualization(model, save_dir="runs/gradcam"):
    """可视化卷积核权重"""
    os.makedirs(save_dir, exist_ok=True)

    # Conv1 权重 (6, 1, 5, 5)
    conv1_weights = model.backbone.conv1.weight.data.cpu().numpy()

    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    fig.suptitle('Conv1 卷积核可视化 (6个5x5卷积核)', fontsize=14)

    for i in range(6):
        im = axes[i].imshow(conv1_weights[i, 0], cmap='coolwarm')
        axes[i].set_title(f'Kernel {i+1}', fontsize=11)
        axes[i].axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/conv1_kernels.png", dpi=150)
    plt.close()
    print(f"✓ Conv1卷积核图已保存: {save_dir}/conv1_kernels.png")

    # Conv2 权重 (16, 6, 5, 5) - 取每个卷积核的mean
    conv2_weights = model.backbone.conv2.weight.data.cpu().numpy()
    conv2_mean = conv2_weights.mean(axis=1)  # (16, 5, 5)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Conv2 卷积核可视化 (16个5x5卷积核, 6通道mean)', fontsize=14)

    for i in range(16):
        im = axes[i // 4, i % 4].imshow(conv2_mean[i], cmap='coolwarm')
        axes[i // 4, i % 4].set_title(f'Kernel {i+1}', fontsize=10)
        axes[i // 4, i % 4].axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/conv2_kernels.png", dpi=150)
    plt.close()
    print(f"✓ Conv2卷积核图已保存: {save_dir}/conv2_kernels.png")


def main():
    device = get_device()
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model("runs/best.pt", device)
    print("✓ 模型加载成功")

    # 1. Grad-CAM 全数字热力图
    print("\n" + "=" * 50)
    print("1. 生成 Grad-CAM 类激活图...")
    plot_gradcam_all_digits(model, device)

    # 2. 梯度分布
    print("\n" + "=" * 50)
    print("2. 生成梯度分布图...")
    plot_gradient_distribution(model, device)

    # 3. 卷积核可视化
    print("\n" + "=" * 50)
    print("3. 生成卷积核可视化...")
    plot_conv_kernel_visualization(model)

    print("\n" + "=" * 50)
    print("所有 Grad-CAM 可视化图已生成完成!")
    print("保存目录: runs/gradcam/")
    print("=" * 50)


if __name__ == "__main__":
    main()
