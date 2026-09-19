import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from model.model import LeNet5


def get_device():
    """鑾峰彇璁粌璁惧: 浼樺厛 GPU, 娌℃湁灏辩敤 CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dataloaders(data_root="./data", batch_size=64, val_ratio=0.1):
    """
    鍔犺浇 MNIST 鏁版嵁闆?    杞崲: 缂╂斁鍒?32x32 (LeNet-5 杈撳叆瑕佹眰), 褰掍竴鍖?    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 璁粌闆?+ 楠岃瘉闆?    full_train = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )

    # 鍒掑垎璁粌闆嗗拰楠岃瘉闆?    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # 娴嬭瘯闆?    test_set = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """璁粌涓€涓?epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 鍓嶅悜浼犳挱
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 鍙嶅悜浼犳挱 + 鏇存柊
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 缁熻
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """璇勪及妯″瀷"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path="runs/training_curves.png"):
    """缁樺埗 loss 鍜?accuracy 鏇茬嚎"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 鏇茬嚎
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 鏇茬嚎
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"璁粌鏇茬嚎宸蹭繚瀛? {save_path}")


def main():
    # 瓒呭弬鏁?    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    # 璁惧
    device = get_device()
    print(f"浣跨敤璁惧: {device}")

    # 鏁版嵁
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # 妯″瀷
    model = LeNet5(num_classes=10).to(device)
    print(f"妯″瀷缁撴瀯:\n{model}")

    # 鎹熷け鍑芥暟 + 浼樺寲鍣?    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 璁板綍璁粌杩囩▼
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    # 鍒涘缓淇濆瓨鐩綍
    os.makedirs("runs", exist_ok=True)

    # 璁粌寰幆
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)

        # 璁粌
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # 楠岃瘉
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # 璁板綍
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 淇濆瓨鏈€浣虫ā鍨?        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "runs/best.pt")
            print(f"鉁?淇濆瓨鏈€浣虫ā鍨?(Val Acc: {best_val_acc:.2f}%)")

    # 淇濆瓨鏈€鍚庝竴杞ā鍨?    torch.save(model.state_dict(), "runs/last.pt")
    print(f"\n鉁?淇濆瓨鏈€鍚庝竴杞ā鍨? runs/last.pt")

    # 缁樺埗璁粌鏇茬嚎
    plot_curves(train_losses, val_losses, train_accs, val_accs)

    # 娴嬭瘯闆嗚瘎浼?    print("\n" + "=" * 50)
    print("娴嬭瘯闆嗚瘎浼?")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print(f"鏈€浣抽獙璇佸噯纭巼: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()



import torch

# 鍔犺浇妯″瀷鏉冮噸
state_dict = torch.load("runs/best.pt")

# 鎵撳嵃鎵€鏈夊眰鐨勫悕瀛楀拰褰㈢姸
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
