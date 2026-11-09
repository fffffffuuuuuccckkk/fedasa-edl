import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# --- 1. 配置参数 (根据您的实际情况设置) ---
TARGET_GPU_INDEX = 2  # 您的模型和数据所在且出错的 GPU 索引
TEST_BATCH_SIZE = 16  # 初始测试 Batch Size，如果失败，请尝试 8 或 4
INPUT_CHANNELS = 3    # 假设是 RGB 图像输入
OUTPUT_CHANNELS = 16  # 任意输出通道数
KERNEL_SIZE = 3       # 常用卷积核大小
INPUT_SIZE = 32       # 假设输入图像是 32x32

# --- 2. 诊断函数 ---

def check_tensor_health(tensor, name):
    """检查张量是否包含 NaN/Inf，并打印设备信息"""
    if not tensor.is_cuda:
        print(f"[{name}] ERROR: 张量不在 CUDA 上，而是在 {tensor.device} 上。")
        return False

    tensor_cpu = tensor.cpu()
    has_nan = torch.isnan(tensor_cpu).any()
    has_inf = torch.isinf(tensor_cpu).any()

    print(f"[{name}] 设备: {tensor.device}, 形状: {list(tensor.shape)}, dtype: {tensor.dtype}")
    print(f"[{name}] 包含 NaN: {has_nan.item()}, 包含 Inf: {has_inf.item()}")

    if has_nan or has_inf:
        print(f"[{name}] 警告：张量包含不稳定的数值！")
        return False
    return True

# --- 3. 核心测试函数 ---

def test_conv_layer():
    print(f"--- PyTorch 卷积层诊断开始 (目标 GPU: cuda:{TARGET_GPU_INDEX}) ---")

    if not torch.cuda.is_available():
        print("致命错误：CUDA 不可用！请检查 PyTorch 环境。")
        return

    # 设置目标 GPU 为当前设备 (解决上下文不一致问题)
    try:
        if torch.cuda.current_device() != TARGET_GPU_INDEX:
            torch.cuda.set_device(TARGET_GPU_INDEX)
            print(f"✅ 当前 CUDA 上下文已设置为 cuda:{TARGET_GPU_INDEX}")
        else:
            print(f"ℹ️ 当前 CUDA 上下文已经是 cuda:{TARGET_GPU_INDEX}")
    except Exception as e:
        print(f"❌ 设置设备失败: {e}")
        return

    # 清理 GPU 缓存，释放内存
    torch.cuda.empty_cache()
    print("🧹 已清理 CUDA 缓存。")

    # 1. 模拟输入数据 (在目标 GPU 上创建)
    # 使用随机数据进行测试
    x = torch.randn(
        TEST_BATCH_SIZE,
        INPUT_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
        dtype=torch.float32
    ).to(f'cuda:{TARGET_GPU_INDEX}')

    # 2. 模拟卷积模块 (在目标 GPU 上创建)
    try:
        conv_layer = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=1,
            bias=True
        ).to(f'cuda:{TARGET_GPU_INDEX}')
    except Exception as e:
        print(f"❌ 卷积层初始化失败: {e}")
        return

    # 3. 检查数据和模型状态
    print("\n--- 状态检查 ---")
    check_tensor_health(x, "Input (x)")
    check_tensor_health(conv_layer.weight, "Conv Weight")

    # 4. 执行卷积操作
    print(f"\n--- 执行 nn.Conv2d Forward (Batch Size: {TEST_BATCH_SIZE}) ---")
    try:
        output = conv_layer(x)
        print("✅ 测试成功！nn.Conv2d 顺利执行。")
        check_tensor_health(output, "Output")

    except RuntimeError as e:
        if "CUBLAS_STATUS_EXECUTION_FAILED" in str(e):
            print("❌ 致命错误：CUBLAS_STATUS_EXECUTION_FAILED 再次发生！")
            print("这确认了问题与您的代码逻辑无关，是该 GPU 或其驱动程序状态不稳定。")
            print(f"建议尝试：1. 再次降低 TEST_BATCH_SIZE。 2. 联系管理员重启服务器。")
        else:
            print(f"❌ 卷积执行失败，但不是 CUBLAS 错误: {e}")

    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

if __name__ == "__main__":
    test_conv_layer()