# S1-Base-API

## 项目介绍

本项目为 S1-Base 系列中的通用科学大语言模型API，该API使用了S1-Base科学大模型，该模型系统地学习并理解 “数理化天地生” 六大基础学科核心理论、定律与专业知识，依托 1.7 亿篇科研论文，在数百万条高质量科学推理数据上经过科学指令微调和多学科复合奖励强化学习训练得到，并通过高中、本科及硕博课程式训练策略逐步强化其学科能力。该系列模型能够根据用户问题自动路由各个领域专用模型，包括谱、场、分子材料、生物领域Evo2、生物领域ESM3、生物领域 AlphaFold2。

该模型共有三个参数量级，分别是 S1-Base-8B，S1-Base-32B 和 S1-Base-671B，其中 S1-Base-8B 和 S1-Base-32B 分别基于 [Qwen3-8B](https://github.com/QwenLM/Qwen3) 和 [Qwen3-32B](https://github.com/QwenLM/Qwen3) 训练得到，S1-Base-671B 基于 [DeepSeek-R1-671B](https://github.com/deepseek-ai/DeepSeek-R1) 训练得到，均支持 32k 上下文。

## 基座模型权重

S1-Base 模型以 Apache 2.0 协议开源，您可以在 [Huggingface](https://huggingface.co/collections/ScienceOne-AI/s1-base-687a2373fde4791bc6c761f0) 或 [ModelScope](https://modelscope.cn/collections/S1-Base-66b70cf6e51c48) 下载模型权重。

| 模型名称     | Huggingface地址                                                   | ModelScope地址                                                          |
| ------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| S1-Base-8B   | [S1-Base-8B](https://huggingface.co/ScienceOne-AI/S1-Base-8B)     | [S1-Base-8B](https://modelscope.cn/models/ScienceOne-AI/S1-Base-8B)     |
| S1-Base-32B  | [S1-Base-32B](https://huggingface.co/ScienceOne-AI/S1-Base-32B)   | [S1-Base-32B](https://modelscope.cn/models/ScienceOne-AI/S1-Base-32B)   |
| S1-Base-671B | [S1-Base-671B](https://huggingface.co/ScienceOne-AI/S1-Base-671B) | [S1-Base-671B](https://modelscope.cn/models/ScienceOne-AI/S1-Base-671B) |

## 项目结构

```
├── api/                 # 基础 API 服务
├── services/            # 各功能服务模块
│   ├── esm3/        
│   ├── field/       
│   ├── mattergen/   
│   └── spectrum/    
├── docker-compose.yml   
└── README.md
```

## 服务列表

1. **alphafold2-multimer**：基于 DeepMind 的 AlphaFold2 Multimer 模型。
   1. [相关文档](https://build.nvidia.com/deepmind/alphafold2-multimer/deploy)
2. **evo2**：基于 `nvcr.io/nim/arc/evo2-40b` 镜像。
   1. [相关文档](https://build.nvidia.com/arc/evo2-40b/deploy)
3. **mattergen**：从本地 `./services/mattergen` 构建的分子材料领域服务。
   1. [相关文档](https://github.com/microsoft/mattergen)
4. **esm3**：从本地 `./services/esm3` 构建的蛋白质序列服务。
   1. [相关文档](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1)
5. **spectrum**：从本地 `./services/spectrum` 构建的谱数据相关服务。
   1. adapter/checkpoint/optimizer.pt[下载地址](https://scienceone.ia.ac.cn/oss/s1-base-spectrum-adapter-checkpoint/optimizer.pt)
   2. adapter/checkpoint/adapter_model.safetensors[下载地址](https://scienceone.ia.ac.cn/oss/s1-base-spectrum-adapter-checkpoint/adapter_model.safetensors)
   3. adapter/adapter_model.safetensors[下载地址](https://scienceone.ia.ac.cn/oss/s1-base-spectrum-adapter/adapter_model.safetensors)
6. **field**：从本地 `./services/field` 构建的场数据相关服务。
   1. [文件下载](https://scienceone.ia.ac.cn/oss/s1-base-field-base-model/base_model.pth)
7. **s1-base-api**：从本地 `./api` 构建的基础 API 服务，作为整个系统的入口。
   1. [相关文档](https://huggingface.co/collections/ScienceOne-AI/s1-base-687a2373fde4791bc6c761f0)

## 环境配置

1. 安装依赖

```bash
# 安装Docker和Docker Compose
brew install docker docker-compose

# 安装NVIDIA Container Toolkit
brew install nvidia-docker
```

2. 修改docker-compose.yml相关环境变量
3. 修改docker-compose.yml中的服务配置

## 服务管理

### 构建和启动服务

```bash
# 构建所有服务镜像
docker-compose build

# 启动所有服务（后台运行）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f [service_name]
```

### 常用命令

```bash
# 停止所有服务
docker-compose down

# 停止并删除容器、镜像
docker-compose down --rmi all

# 重启特定服务
docker-compose restart esm3
```

## API 使用示例

### 蛋白质序列补全

```bash
curl --location --request POST 'http://127.0.0.1:8000/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-raw '{
      "model": "S1-Base",
      "messages": [
         {
           "role": "user",
           "content": "我有一段蛋白质序列：QATSLRILNNGHAFNVEFDDSQDKAVL，请帮我补全这个蛋白质序列，序列左侧需要补全的长度为50，序列右侧需要补全的长度为50。"
         }
     ]
}'
```

### response

```json
{ 
  "id": "chatcmpl-1752742228582",
  "object": "chat.completion",
  "created": 1752742228,
  "model": "S1-Base",
  "choices": [
    { 
      "index": 0,
      "message":
        { 
          "role": "assistant",
          "content": "以下是补全后的蛋白质序列：\n\n```\nMKLALFIVALFHQLYGVADGTRQTNKCCHQVSVQQVDDAGIQLLSIIFSFQATSLRILNNGHAFNVEFDDSQDKAVLQVFDQQDQLLTIIKAVATLQDGDQLTIHGIQFQITEVDSSAQITLSLIHI\n```\n\n其中：\n- 原始序列：`QATSLRILNNGHAFNVEFDDSQDKAVL`\n- 左侧补全：`MKLALFIVALFHQLYGVADGTRQTNKCCHQVSVQQVDDAGIQLLSIIFSF`\n- 右侧补全：`QVFDQQDQLLTIIKAVATLQDGDQLTIHGIQFQITEVDSSAQITLSLIHI`\n\n补全成功！"
       },
     "finish_reason": "stop"
    }
  ],
  "usage":
    {
      "completion_tokens": 253,
      "prompt_tokens": 1055,
      "total_tokens": 1308
    }
}
```



