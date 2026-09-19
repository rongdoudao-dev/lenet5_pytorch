import torch
from model.model import LeNet5


def check_model():
    """查看模型权重内容"""
    # 创建模型
    model = LeNet5(num_classes=10)

    # 加载最佳权重
    model.load_state_dict(torch.load("runs/best.pt", map_location="cpu"))

    # 打印模型结构
    print("=" * 60)
    print("模型结构:")
    print("=" * 60)
    print(model)

    print("\n" + "=" * 60)
    print("各层权重形状:")
    print("=" * 60)

    # 遍历所有参数
    for name, param in model.named_parameters():
        print(f"{name:30s} | 形状: {str(param.shape):20s} | 参数量: {param.numel():>8d}")

    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 测试一下前向传播
    print("\n" + "=" * 60)
    print("测试前向传播:")
    print("=" * 60)
    dummy_input = torch.randn(1, 1, 32, 32)  # 假的输入图
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"预测结果: 数字 {torch.argmax(output).item()}")


if __name__ == "__main__":
    check_model()
