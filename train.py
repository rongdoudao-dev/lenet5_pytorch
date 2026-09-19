import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from model.model import LeNet5


def get_device():
    """获取训练设备: 优先 GPU, 没有就用 CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dataloaders(data_root="./data", batch_size=64, val_ratio=0.1):
    """
    加载 MNIST 数据集
    转换: 缩放到 32x32 (LeNet-5 输入要求), 归一化
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        batch_losses.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, batch_losses


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, classes, title, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def compute_confusion_matrix(preds, labels, num_classes=10):
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_pr_curve(all_labels, all_probs, num_classes=10, save_path="runs/pr_curve.png"):
    """绘制 PR 曲线"""
    plt.figure(figsize=(10, 6))

    for i in range(num_classes):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]

        sorted_indices = np.argsort(class_probs)[::-1]
        sorted_labels = binary_labels[sorted_indices]
        sorted_probs = class_probs[sorted_indices]

        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        fn = tp[-1] - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        plt.plot(recall, precision, label=f'Class {i}', alpha=0.7)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (One-vs-Rest)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PR曲线已保存: {save_path}")


def plot_predictions(model, test_loader, device, save_path="runs/predictions.png"):
    """绘制预测样例"""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    axes = axes.ravel()

    for i in range(24):
        img = images[i].squeeze().numpy()
        img = img * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f'T:{labels[i]} P:{predicted[i]}', color=color, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"预测样例已保存: {save_path}")


def plot_gradient_monitoring(model, save_path="runs/gradient_monitoring.png"):
    """梯度监控图"""
    layers = []
    grad_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layers.append(name)
            grad_norms.append(param.grad.norm().item())

    plt.figure(figsize=(12, 6))
    plt.barh(layers, grad_norms)
    plt.xlabel('Gradient L2 Norm')
    plt.title('Gradient Monitoring')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"梯度监控已保存: {save_path}")


def plot_learning_rate(lr_history, save_path="runs/learning_rate.png"):
    """学习率曲线"""
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(lr_history) + 1), lr_history, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"学习率曲线已保存: {save_path}")


def plot_training_dashboard(train_losses, val_losses, train_accs, val_accs,
                            batch_losses_all, save_path="runs/training_dashboard.png"):
    """训练仪表盘"""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy曲线
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Acc')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Batch Loss
    axes[1, 0].plot(batch_losses_all, 'g-', linewidth=0.5)
    axes[1, 0].set_title('Batch Loss (Every Iteration)')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Accuracy Gap
    gap = np.array(train_accs) - np.array(val_accs)
    axes[1, 1].plot(epochs, gap, 'm-')
    axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap (%)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练仪表盘已保存: {save_path}")


def plot_management_dashboard(train_acc, val_acc, test_acc, train_loss, val_loss, test_loss,
                               best_val_acc, total_epochs, save_path="runs/management_dashboard.png"):
    """管理层仪表盘"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 准确率对比
    metrics = ['Train', 'Val', 'Test']
    values = [train_acc, val_acc, test_acc]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    axes[0].bar(metrics, values, color=colors)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy Overview')
    axes[0].set_ylim([0, 100])
    for i, v in enumerate(values):
        axes[0].text(i, v + 1, f'{v:.2f}%', ha='center')

    # 2. Loss对比
    loss_values = [train_loss, val_loss, test_loss]
    axes[1].bar(metrics, loss_values, color=colors)
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss Overview')
    for i, v in enumerate(loss_values):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center')

    # 3. 关键指标卡片
    axes[2].axis('off')
    info_text = f"""
    Model Performance Summary
    ═══════════════════════
    Epochs: {total_epochs}
    Best Val Acc: {best_val_acc:.2f}%
    
    Train Acc: {train_acc:.2f}%
    Val Acc:   {val_acc:.2f}%
    Test Acc:  {test_acc:.2f}%
    
    Train Loss: {train_loss:.4f}
    Val Loss:   {val_loss:.4f}
    Test Loss:  {test_loss:.4f}
    """
    axes[2].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                 verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"管理层仪表盘已保存: {save_path}")


def main():
    # 超参数
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    # 设备
    device = get_device()
    print(f"使用设备: {device}")

    # 数据
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # 模型
    model = LeNet5(num_classes=10).to(device)
    print(f"模型结构:\n{model}")

    # 损失函数 + 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    batch_losses_all = []
    lr_history = []
    best_val_acc = 0.0

    os.makedirs("runs", exist_ok=True)

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc, batch_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        batch_losses_all.extend(batch_losses)

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        lr_history.append(optimizer.param_groups[0]['lr'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "runs/best.pt")
            print(f"✓ 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")

    # 保存最后一轮模型
    torch.save(model.state_dict(), "runs/last.pt")
    print(f"\n✓ 保存最后一轮模型: runs/last.pt")

    # ===== 生成所有图 =====
    print("\n" + "=" * 50)
    print("生成所有图表...")
    print("=" * 50)

    # 1. 训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, num_epochs + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/training_curves.png", dpi=150)
    plt.close()
    print("✓ training_curves.png")

    # 2. 混淆矩阵（训练集）
    _, _, train_preds, train_labels, _ = evaluate(model, train_loader, criterion, device)
    cm_train = compute_confusion_matrix(train_preds, train_labels)
    plot_confusion_matrix(cm_train, range(10), 'Confusion Matrix - Train Set', 'runs/confusion_train.png')

    # 3. 混淆矩阵（验证集）
    _, _, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device)
    cm_val = compute_confusion_matrix(val_preds, val_labels)
    plot_confusion_matrix(cm_val, range(10), 'Confusion Matrix - Val Set', 'runs/confusion_val.png')

    # 4. 混淆矩阵（测试集）
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )
    cm_test = compute_confusion_matrix(test_preds, test_labels)
    plot_confusion_matrix(cm_test, range(10), 'Confusion Matrix - Test Set', 'runs/confusion_test.png')
    print(f"✓ confusion_test.png")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # 5. PR曲线
    plot_pr_curve(test_labels, test_probs)

    # 6. 预测样例
    plot_predictions(model, test_loader, device)

    # 7. 梯度监控
    plot_gradient_monitoring(model)

    # 8. 学习率曲线
    plot_learning_rate(lr_history)

    # 9. 训练仪表盘
    plot_training_dashboard(train_losses, val_losses, train_accs, val_accs, batch_losses_all)

    # 10. 管理层仪表盘
    plot_management_dashboard(
        train_accs[-1], val_accs[-1], test_acc,
        train_losses[-1], val_losses[-1], test_loss,
        best_val_acc, num_epochs
    )

    print("\n" + "=" * 50)
    print("所有图表生成完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
