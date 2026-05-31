#!/bin/bash
# UAV-MTL-Inference 一键环境配置脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "  UAV-MTL-Inference 环境配置"
echo "========================================="

# 1. 检查Conda
if ! command -v conda &> /dev/null; then
    echo "❌ 错误：未找到Conda，请先安装Anaconda或Miniconda"
    exit 1
fi
echo "✓ 检测到Conda"

# 2. 创建环境
echo ""
echo "正在创建Conda环境 'uav-mtl'..."
if conda env list | grep -q "uav-mtl"; then
    echo "⚠ 环境已存在，将删除并重新创建"
    conda env remove -n uav-mtl -y
fi
conda env create -f environment.yml

# 3. 激活环境并安装依赖
echo ""
echo "正在安装Python依赖..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate uav-mtl
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. 验证安装
echo ""
echo "正在验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import gymnasium; print(f'Gymnasium版本: {gymnasium.__version__}')"

# 5. 创建必要目录
echo ""
echo "正在创建目录结构..."
mkdir -p data/raw data/processed data/pretrained
mkdir -p experiments/accuracy_fitting experiments/training
mkdir -p checkpoints/accuracy_functions
mkdir -p logs/training logs/evaluation
mkdir -p results/figures results/tables

echo ""
echo "========================================="
echo "  ✓ 环境配置完成！"
echo "========================================="
echo ""
echo "使用方法："
echo "  conda activate uav-mtl"
echo "  python scripts/data/0_download_nyuv2.py"
echo ""

