<div align="center"><h1 align="center">S1-Base-API</h1></div>

<div align="center">

[![License](https://img.shields.io/badge/LICENSE-Apache_2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)

</div>

## 项目介绍

本项目为磐石科学基础大模型（S1-Base）API，它采用专业科学知识和数据进行训练，是服务于科学任务的通专融合的科学领域多模态大模型，应用于科学推理场景与复杂学科任务。该模型现阶段采用异构混合专家架构，能够根据用户问题自动“路由”至深度定制的语言大模型或领域专用模型（波、谱、场、蛋白质、生物序列等）。

## 通用大模型

该模型系统地学习并理解 “数理化天地生” 六大基础学科核心理论、定律与专业知识，依托 1.7 亿篇科研论文，在数百万条高质量科学推理数据上经过科学指令微调和多学科复合奖励强化学习训练得到，并通过高中、本科及硕博课程式训练策略逐步强化其学科能力。

该模型共有三个参数量级，分别是 S1-Base-8B，S1-Base-32B 和 S1-Base-671B，其中 S1-Base-8B 和 S1-Base-32B 分别基于 [Qwen3-8B](https://github.com/QwenLM/Qwen3) 和 [Qwen3-32B](https://github.com/QwenLM/Qwen3) 训练得到，S1-Base-671B 基于 [DeepSeek-R1-671B](https://github.com/deepseek-ai/DeepSeek-R1) 训练得到，均支持 32k 上下文。
S1-Base 模型以 Apache 2.0 协议开源，您可以在 [Huggingface](https://huggingface.co/collections/ScienceOne-AI/s1-base-687a2373fde4791bc6c761f0) 或 [ModelScope](https://modelscope.cn/collections/S1-Base-66b70cf6e51c48) 下载模型权重。

| 模型名称     | Huggingface地址                                                   | ModelScope地址                                                          |
| ------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| S1-Base-8B   | [S1-Base-8B](https://huggingface.co/ScienceOne-AI/S1-Base-8B)     | [S1-Base-8B](https://modelscope.cn/models/ScienceOne-AI/S1-Base-8B)     |
| S1-Base-32B  | [S1-Base-32B](https://huggingface.co/ScienceOne-AI/S1-Base-32B)   | [S1-Base-32B](https://modelscope.cn/models/ScienceOne-AI/S1-Base-32B)   |
| S1-Base-671B | [S1-Base-671B](https://huggingface.co/ScienceOne-AI/S1-Base-671B) | [S1-Base-671B](https://modelscope.cn/models/ScienceOne-AI/S1-Base-671B) |

## 领域专业模型

1. alphafold2-multimer: AlphaFold2 是由谷歌旗下人工智能研究实验室 DeepMind 的研究团队开发的一款用于蛋白质结构预测的深度学习模型。AlphaFold2 在其前身 AlphaFold 的成功基础上发展而来，是蛋白质结构预测领域的一项重大突破。
2. evo2: Evo2 是一种生物学基础模型，能够整合长基因组序列上的信息，同时对单核苷酸变化保持敏感性。该模型拥有 400 亿个参数，能理解所有生命领域的遗传密码，是迄今为止规模最大的生物学人工智能模型。Evo 2 的训练数据集包含近 9 万亿个核苷酸。
3. esm3: esm3-sm-open-v1 是在 27.8 亿种天然蛋白质的基础上训练而成的。通过合成数据增强，训练数据涵盖了 31.5 亿条蛋白质序列、2.36 亿个蛋白质结构以及 5.39 亿种带有功能注释的蛋白质，总计达 7710 亿个标记。esm3-sm-open-v1 是一种生成式模型，能够根据序列、结构和功能的部分提示来设计蛋白质。
4. mattergen: MatterGen 是一个用于无机材料设计的生成模型，它可以在整个周期表范围内进行材料设计，并通过微调来满足广泛的属性约束。

## 科学模态模型

1. 场模态模型：计算高铁模型在多种流体环境下的表面压力场，为高铁构型设计提供了数据支持，能够显著提升高铁空气动力学的计算效率，大幅缩短高铁的设计周期。
2. 谱模态模型：实现谱信号到分子微观结构的解析，涵盖质谱、红外光谱、紫外-可见光谱、拉曼光谱、核磁共振谱等谱模态，各模态相对当前水平平均提升31%，最高相对提升67%。



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
   2. [minio安装](https://github.com/minio/minio)
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
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

sudo apt-get install nvidia-docker
```

2. 修改docker-compose.yml相关环境变量
   1. alphafold2-multimer服务的NGC\_CLI\_API\_KEY
   2. evo2服务的NGC\_API\_KEY
   3. mattergen服务的minio配置
3. 修改docker-compose.yml中的服务配置
   1. 服务启动的端口
   2. 服务部署的GPU序号

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

## 📄 开源协议

本项目基于 [Apache-2.0 License](LICENSE) 开源发布，欢迎学术与商业使用。

## 🤝 致谢

本项目基于[vllm](https://github.com/vllm-project/vllm)、[langgraph](https://github.com/langchain-ai/langgraph)、[minio](https://github.com/minio/minio)等开源项目以及[Qwen3](https://qwenlm.github.io/blog/qwen3/)、[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[alphafold2-multimer](https://build.nvidia.com/deepmind/alphafold2-multimer/deploy)、[esm3](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1)、[evo2](https://build.nvidia.com/arc/evo2-40b)等开源模型和领域专用模型。感谢所有开源社区的贡献！

