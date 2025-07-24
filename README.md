# soneBaseAPI

一个集成多种生物信息学和深度学习服务的基础 API 平台，提供蛋白质结构预测、序列分析、分子生成等功能。

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

1. **alphafold2-multimer**：基于 DeepMind 的 AlphaFold2 Multimer 模型，提供蛋白质结构预测服务。
   1. 相关文档：https://build.nvidia.com/deepmind/alphafold2-multimer/deploy
2. **evo2**：使用 `nvcr.io/nim/arc/evo2-40b` 镜像，提供相关的深度学习服务。
   1. 相关文档：https://build.nvidia.com/arc/evo2-40b/deploy
3. **mattergen**：从本地 `./services/mattergen` 构建的服务，用于分子生成任务。
   1. 相关文档: https://github.com/microsoft/mattergen
4. **esm3**：从本地 `./services/esm3` 构建的服务，用于蛋白质序列分析。
   1. 相关文档：https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
5. **spectrum**：从本地 `./services/spectrum` 构建的服务，提供光谱分析功能。
   1. adapter/checkpoint/optimizer.pt下载地址:https://scienceone.ia.ac.cn/oss/s1-base-spectrum-adapter-checkpoint/optimizer.pt
   2. adapter/checkpoint/adapter_model.safetensors下载地址:https://scienceone.ia.ac.cn/oss/s1-base-spectrum-adapter-checkpoint/adapter_model.safetensors
   3. adapter/adapter_model.safetensors下载地址： https://scienceone.ia.ac.cn/oss/s1-base-spectrum-adapter/adapter_model.safetensors
6. **field**：从本地 `./services/field` 构建的服务，提供领域特定功能。
   1. 模型下载：https://scienceone.ia.ac.cn/oss/s1-base-field-base-model/base_model.pth
7. **sone-base**：从本地 `./api` 构建的基础 API 服务，作为整个系统的入口。
   1. 相关文档：https://huggingface.co/collections/ScienceOne-AI/s1-base-687a2373fde4791bc6c761f0

## 前提条件

- Docker Engine (20.10+) 和 Docker Compose (v2.0+)
- NVIDIA GPU
- NVIDIA Container Toolkit

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

## 故障排除

1. **服务启动失败**
   
   - 检查端口是否被占用：`lsof -i :8000`
   - 验证GPU是否可用：`nvidia-smi`

## 常见问题

- Q: 哪些服务不需要GPU？
  A: mattergen和spectrum服务可以在CPU上运行，但速度会显著降低
- Q: 如何更新服务版本？
  A: 拉取最新代码后执行`docker-compose build --no-cache`

