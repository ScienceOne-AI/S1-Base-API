# 安装系统依赖
# 使用NVIDIA官方CUDA镜像作为基础镜像
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=7

# 安装系统依赖和Python 3.10
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 升级pip到最新版本
RUN python3 -m pip install --upgrade pip setuptools wheel

# 设置工作目录
WORKDIR /app

COPY . /app

# 分步安装依赖以避免冲突
RUN pip install torch==2.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install transformers>=4.48.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install peft==0.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install gunicorn==21.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install llamafactory==0.9.3 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install huggingface-hub -i https://mirrors.aliyun.com/pypi/simple/


ENV HF_ENDPOINT=https://hf-mirror.com
RUN huggingface-cli download Qwen/Qwen3-32B --local-dir ./Qwen3-32B


# 暴露端口
EXPOSE 5002

# 启动命令
CMD ["llamafactory-cli", "api", "api_lora.yaml"]