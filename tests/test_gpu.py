"""
GPU功能测试脚本
测试PyTorch CUDA环境是否正确配置，以及GPU是否可用于训练
"""

import torch
import sys
from pathlib import Path


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_pytorch_installation():
    """测试PyTorch安装"""
    print_section("PyTorch 安装检查")
    
    try:
        import torch
        import torchvision
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ TorchVision版本: {torchvision.__version__}")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_cuda_availability():
    """测试CUDA是否可用"""
    print_section("CUDA 可用性检查")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: True")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        return True
    else:
        print("✗ CUDA不可用!")
        print("\n可能的原因:")
        print("1. PyTorch安装的是CPU版本")
        print("2. NVIDIA驱动未正确安装")
        print("3. CUDA toolkit版本不匹配")
        print("\n解决方法:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False


def test_gpu_info():
    """测试GPU详细信息"""
    print_section("GPU 详细信息")
    
    if not torch.cuda.is_available():
        print("✗ 跳过（CUDA不可用）")
        return False
    
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        props = torch.cuda.get_device_properties(i)
        print(f"名称: {torch.cuda.get_device_name(i)}")
        print(f"计算能力: {props.major}.{props.minor}")
        print(f"总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"多处理器数量: {props.multi_processor_count}")
        print(f"最大线程数/块: {props.max_threads_per_multi_processor}")
        
        # 当前显存使用情况
        print(f"\n当前显存使用:")
        print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.3f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved(i) / 1024**3:.3f} GB")
        print(f"  可用: {(props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3:.2f} GB")
    
    return True


def test_basic_operations():
    """测试基本GPU操作"""
    print_section("GPU 基本操作测试")
    
    if not torch.cuda.is_available():
        print("✗ 跳过（CUDA不可用）")
        return False
    
    try:
        # 测试1: 创建tensor
        print("\n测试1: 创建GPU tensor...")
        x = torch.randn(1000, 1000, device='cuda')
        print(f"✓ 成功创建shape为{x.shape}的tensor")
        
        # 测试2: 矩阵乘法
        print("\n测试2: GPU矩阵乘法...")
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.mm(x, y)
        print(f"✓ 矩阵乘法成功，结果shape: {z.shape}")
        
        # 测试3: CPU-GPU数据传输
        print("\n测试3: CPU ↔ GPU 数据传输...")
        cpu_tensor = torch.randn(500, 500)
        gpu_tensor = cpu_tensor.cuda()
        back_to_cpu = gpu_tensor.cpu()
        print(f"✓ 数据传输成功")
        
        # 清理显存
        del x, y, z, gpu_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ 操作失败: {e}")
        return False


def test_training_simulation():
    """测试简单训练循环"""
    print_section("训练循环模拟测试")
    
    if not torch.cuda.is_available():
        print("✗ 跳过（CUDA不可用）")
        return False
    
    try:
        print("\n创建简单神经网络...")
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"✓ 模型已移至GPU")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n执行10次前向/反向传播...")
        for i in range(10):
            # 模拟batch
            inputs = torch.randn(32, 100, device='cuda')
            targets = torch.randint(0, 10, (32,), device='cuda')
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 5 == 0:
                print(f"  迭代 {i+1}/10, Loss: {loss.item():.4f}")
        
        print("✓ 训练循环模拟成功")
        
        # 显示显存使用
        print(f"\n训练后显存使用:")
        print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  峰值: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        
        # 清理
        del model, optimizer, criterion, inputs, targets, outputs, loss
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_precision():
    """测试混合精度训练"""
    print_section("混合精度训练测试 (FP16)")
    
    if not torch.cuda.is_available():
        print("✗ 跳过（CUDA不可用）")
        return False
    
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        print("\n创建模型并使用混合精度...")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10)
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()
        criterion = torch.nn.CrossEntropyLoss()
        
        print("✓ 混合精度训练器已创建")
        
        # 模拟训练
        inputs = torch.randn(8, 3, 32, 32, device='cuda')
        targets = torch.randint(0, 10, (8,), device='cuda')
        
        # 混合精度前向传播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 混合精度反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✓ 混合精度训练成功, Loss: {loss.item():.4f}")
        print(f"  显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print("\n提示: 使用混合精度可以节省约40%显存，加速训练约2-3倍")
        
        # 清理
        del model, optimizer, scaler, criterion, inputs, targets, outputs, loss
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ 混合精度测试失败: {e}")
        return False


def test_batch_size_recommendation():
    """测试并推荐batch size"""
    print_section("Batch Size 推荐")
    
    if not torch.cuda.is_available():
        print("✗ 跳过（CUDA不可用）")
        return False
    
    try:
        # 获取可用显存
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n总显存: {total_memory:.2f} GB")
        
        print("\n推荐配置（NYUv2数据集，图像288x288）:")
        print("-" * 60)
        
        if total_memory >= 8:
            print("MTAN 模型:")
            print("  • Batch size: 6-8")
            print("  • 预计显存占用: 5-6 GB")
            print("  • 推荐使用混合精度: 可选")
            
            print("\nSplit Network 模型:")
            print("  • Batch size: 8-12")
            print("  • 预计显存占用: 4-5 GB")
            print("  • 推荐使用混合精度: 可选")
            
            print("\nCross-Stitch Networks:")
            print("  • Batch size: 4-6")
            print("  • 预计显存占用: 6-7 GB")
            print("  • 推荐使用混合精度: 建议")
        else:
            print("⚠ 显存较小，建议:")
            print("  • 降低batch size到2-4")
            print("  • 必须使用混合精度训练")
            print("  • 考虑降低图像分辨率")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "█" * 70)
    print("  GPU 功能完整测试")
    print("  NVIDIA RTX 2000 Ada Generation Laptop GPU")
    print("█" * 70)
    
    results = {}
    
    # 运行所有测试
    results['PyTorch安装'] = test_pytorch_installation()
    results['CUDA可用性'] = test_cuda_availability()
    
    if results['CUDA可用性']:
        results['GPU信息'] = test_gpu_info()
        results['基本操作'] = test_basic_operations()
        results['训练模拟'] = test_training_simulation()
        results['混合精度'] = test_mixed_precision()
        results['Batch Size推荐'] = test_batch_size_recommendation()
    
    # 打印测试总结
    print_section("测试总结")
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n" + "█" * 70)
        print("  🎉 所有测试通过！GPU环境配置正确！")
        print("  可以开始训练多任务模型了！")
        print("█" * 70)
    else:
        print("\n" + "█" * 70)
        print("  ⚠ 部分测试失败，请检查环境配置")
        print("█" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

