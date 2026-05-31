# UAV-MTL-Inference Docker镜像
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt requirements-dev.txt ./

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口（Tensorboard, Jupyter）
EXPOSE 6006 8888

# 默认命令
CMD ["/bin/bash"]

