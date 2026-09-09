# Knowledge Forge 换机部署手册

本文是当前版本的换机部署入口。部署时只需要迁移源代码、配置、业务数据和模型服务的连接信息；不要把 `.env`、上传文件、向量文件、模型权重或运行日志提交到 Git。

## 1. 先选择运行模式

Knowledge Forge 的模型调用分为两个独立通道：

| 通道 | 默认实现 | 可替代实现 | 用途 |
| --- | --- | --- | --- |
| LLM | Ollama | OpenAI 兼容 Chat Completions | 结构化抽取、Schema、社区报告、问答生成 |
| Embedding | Ollama | OpenAI 兼容 Embeddings | 知识块、实体和查询向量 |
| Reranker | 独立 HTTP 服务 | 远程 reranker 或关闭 | 实体归一和问答候选重排 |

有 GPU 的机器推荐使用根目录 `docker-compose.yml`。没有 GPU 时有两种选择：

1. 实用方案：应用容器使用 CPU Compose，LLM 和 Embedding 通过远程 OpenAI 兼容 API 提供；reranker 使用远程服务或关闭。
2. 纯本地方案：使用 CPU Compose，并在本机运行 Ollama CPU。可以工作，但抽取、社区报告和多模态解析会明显变慢，不适合大规模首次构建。

`docker-compose.cpu.yml` 只替换 GPU Python 依赖并移除 GPU 请求，不改变业务流程、数据库格式或图谱策略。

## 2. 安装基础软件

以 Ubuntu/Debian 为例，安装 Git、Docker Compose 插件、curl 和 Python 3.12：

```bash
sudo apt-get update
sudo apt-get install -y git curl python3.12 python3.12-venv
```

Docker 请按官方文档安装 Docker Engine 和 Compose v2。GPU 机器还需要 NVIDIA 驱动和 NVIDIA Container Toolkit，并用下面命令确认容器可见 GPU：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

参考：

- [Docker Engine 安装](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 3. 获取代码和创建配置

当前 `main` 是可部署分支：

```bash
git clone <仓库地址> /opt/knowledge-forge
cd /opt/knowledge-forge
git switch main
cp .env.docker.example .env.docker
```

编辑 `.env.docker`：

- `POSTGRES_PASSWORD` 改为随机密码；
- `SERVICE_HOST` 和 `HTTP_BASE_URL` 填浏览器能够访问的地址；只在确有局域网访问需求时使用非回环地址；
- `HUGEGRAPH_HOST/HUGEGRAPH_PORT` 指向 HugeGraph；
- 根据第 5 节配置本地 Ollama 或远程 API；
- 根据第 6 节配置 reranker；
- 生产环境必须启用真实认证和防火墙，不要直接沿用示例中的开发模式。

检查 Compose 展开结果：

```bash
docker compose --env-file .env.docker config --quiet
```

## 4. PostgreSQL 与 pgvector

根目录 Compose 会构建带 pgvector 的 PostgreSQL 16，并把数据保存到命名卷 `knowledge-forge-postgres-data`。应用启动时自动创建扩展、迁移和业务表：

```bash
docker compose --env-file .env.docker up -d postgres
docker compose --env-file .env.docker ps
```

检查：

```bash
docker compose --env-file .env.docker exec postgres \
  pg_isready -U postgres -d knowledge_forge
```

不要删除该卷来“解决”迁移问题。换机时用 `pg_dump` 或完整 PostgreSQL 备份迁移：

```bash
docker compose --env-file .env.docker exec -T postgres \
  pg_dump -U postgres -d knowledge_forge > /secure/knowledge_forge.sql
```

pgvector 数据属于 PostgreSQL 业务数据，不存储在 HugeGraph 中。它包括知识块、实体和查询所需的向量索引；GraphRAG Native 的向量则只在独立的 LanceDB bundle 中保存。

## 5. 模型服务

### 5.1 本地 Ollama

在运行 Ollama 的机器上安装并启动服务：

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull qwen3.6:latest
ollama pull bge-m3:latest
curl http://127.0.0.1:11434/api/tags
```

`qwen3.6:latest` 用于 LLM，`bge-m3:latest` 用于 Embedding。Docker 中的应用通过 `host.docker.internal:11434` 访问宿主机 Ollama；本地虚拟环境则使用 `127.0.0.1:11434`。

如果 Ollama 与应用不在同一台机器，需要让 Ollama 监听可信局域网地址，并只在防火墙中允许应用机器访问。不要把未认证的 11434 端口暴露到公网。随后设置：

```dotenv
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://<ollama-host>:11434
OLLAMA_EMBEDDING_BASE_URL=http://<ollama-host>:11434
DEFAULT_LLM_MODEL=qwen3.6:latest
DEFAULT_EMBEDDING_MODEL=bge-m3:latest
```

### 5.2 无 GPU 机器使用远程 LLM/Embedding API

应用不要求远端一定是 OpenAI 官方服务，只要求提供方兼容以下接口：

- `POST /v1/chat/completions`，返回 `choices[0].message.content`；
- `POST /v1/embeddings`，返回 `data[]` 中的 `embedding` 数组；
- 使用 Bearer API Key（如果服务不需要 Key，可以留空）。

在 `.env.docker` 中分别设置通道：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_API_BASE_URL=https://llm.example.com/v1
LLM_API_KEY=<llm-secret>
LLM_API_MODELS=<chat-model-name>
LLM_API_CONTEXT_TOKENS=32768

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_BASE_URL=https://embedding.example.com/v1
EMBEDDING_API_KEY=<embedding-secret>
EMBEDDING_API_MODELS=<embedding-model-name>
EMBEDDING_API_CONTEXT_TOKENS=8192

DEFAULT_LLM_MODEL=<chat-model-name>
DEFAULT_EMBEDDING_MODEL=<embedding-model-name>
```

如果两个模型由同一个网关提供，可以把两个 Base URL 和 Key 填成相同值。`LLM_API_MODELS` 和 `EMBEDDING_API_MODELS` 是逗号分隔的模型名；它们用于项目创建页面，不依赖远端实现 `/v1/models`。首次进入页面也能看到配置的模型，点击同步按钮会把当前提供方模型写入 PostgreSQL。

服务启动后，也可以在主页右上角打开“设置”维护相同的提供方、API 地址、模型列表、上下文
上限和默认模型。主页设置写入 PostgreSQL 的全局 `settings` 表：默认模型只在新建知识库时
作为初始值，已有知识库的模型、语言、上下文预算和构建结果不会被改写。知识库设置页还可
配置当前知识库专用的提供方、API 地址、Key、模型列表、Ollama 地址和上下文上限；留空的
项目字段继承主页全局设置，填写后的项目字段只作用于当前知识库。后台任务会在项目作用域
内建立对应模型连接，不会通过修改全局客户端影响其他知识库。API Key 不会回显到
页面；密码框留空表示保留当前 Key，勾选清除选项才会删除。若 LLM 和 Embedding 都使用远程
API，则本地 Ollama 地址可以留空；选择本地 Ollama 的通道时，才需要填写对应地址。

API 模式下，Knowledge Forge 仍执行相同的结构化 JSON 校验、必填字段校验、失败重试和引用门禁；API 的 `response_format` 只负责 JSON 形状，不能替代服务端 Schema 校验。图片分析要求 LLM 端支持 OpenAI 图片消息格式；不支持视觉输入的 API 只能处理文本。

GraphRAG Native 也读取这两个 API 通道，并将 completion 和 embedding 地址分别写入隔离 run 的 settings。若两个远程服务使用不同 Key，应用会分别注入 `GRAPHRAG_API_KEY` 与 `GRAPHRAG_EMBEDDING_API_KEY`；密钥不会写入 settings 文件。

### 5.3 先远程 LLM、Embedding 仍用本地 CPU Ollama

这也是合法组合：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_API_BASE_URL=https://llm.example.com/v1
LLM_API_KEY=<llm-secret>
LLM_API_MODELS=<chat-model-name>
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_BASE_URL=http://host.docker.internal:11434
DEFAULT_LLM_MODEL=<chat-model-name>
DEFAULT_EMBEDDING_MODEL=bge-m3:latest
```

这种组合能减少本地 LLM 负担，但 Embedding 仍会占用 CPU；向量空间变化时不要直接复用旧向量，应按项目提供的重建/迁移流程重建 Embedding 索引。

## 6. HugeGraph

HugeGraph 是可浏览的物理图，PostgreSQL 才是事实、来源、任务和版本控制面。单机开发/验收可以运行固定版本：

```bash
docker run -d --name hugegraph \
  --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -v hugegraph-data:/hugegraph-server \
  hugegraph/hugegraph:1.7.0
```

等待服务后初始化 Knowledge Forge 的物理图：

```bash
./deploy/init_hugegraph_graph.sh hugegraph hugegraph/hugegraph:1.7.0
curl http://127.0.0.1:8080/graphs
```

结果应包含 `knowledge_forge`。脚本只复制 `deploy/hugegraph/knowledge_forge.properties` 并初始化该物理图，不会删除 `hugegraph` 或 `test`。已有生产实例不要直接执行初始化脚本；应先确认容器名、版本、数据卷和备份。

应用配置：

```dotenv
HUGEGRAPH_HOST=host.docker.internal
HUGEGRAPH_PORT=8080
HUGEGRAPH_GRAPH=knowledge_forge
HUGEGRAPH_AUTH_ENABLED=false
```

生产环境建议开启 HugeGraph 认证、限制网络访问，并填写 `HUGEGRAPH_USER/HUGEGRAPH_PASSWORD`。Hubble 是可选的浏览器管理组件，不是 Knowledge Forge 运行时必需依赖。

参考：[HugeGraph Server Quick Start](https://hugegraph.apache.org/docs/quickstart/hugegraph/hugegraph-server/)。

## 7. Reranker

### 本机独立服务

下载 `BAAI/bge-reranker-v2-m3` 到宿主机，例如：

```bash
mkdir -p /opt/models/bge-reranker-v2-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 \
  --local-dir /opt/models/bge-reranker-v2-m3
```

在 `.env.docker` 中改为：

```dotenv
ENTITY_RERANKER_HOST_PATH=/opt/models/bge-reranker-v2-m3
ENTITY_RERANKER_MODEL_PATH=/models/entity-reranker
ENTITY_RERANKER_URL=http://knowledge-forge-reranker:8012
ENTITY_RERANKER_ENABLED=true
```

GPU Compose 会让 reranker 容器使用 GPU；CPU Compose 会使用 CPU 依赖。CPU 机器也可以关闭它：

```dotenv
ENTITY_RERANKER_ENABLED=false
```

关闭后实体归一和问答检索使用已有的词法、向量、RRF 及确定性降级，不会把服务故障误判为“没有知识”，但候选排序质量可能下降。

### 远程 reranker

如果已有独立 reranker 服务，只需让应用访问其 `/api/rerank` 和 `/health`，并设置：

```dotenv
ENTITY_RERANKER_ENABLED=true
ENTITY_RERANKER_URL=http://<reranker-host>:8012
ENTITY_RERANKER_API_KEY=<reranker-secret>
```

服务端应限制来源 IP 和请求大小；应用发送的是有界候选，不会上传完整数据库。

## 8. 启动应用

### GPU Compose

```bash
docker compose --env-file .env.docker up -d --build
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs -f knowledge-forge
```

### CPU Compose + 远程 API

若不想在本机加载 reranker 模型，使用 API-only CPU 覆盖；它会跳过本地 reranker 容器，保留词法、向量和 RRF 降级排序：

```bash
docker compose --env-file .env.docker \
  -f docker-compose.yml -f docker-compose.cpu.yml -f docker-compose.cpu-api.yml \
  up -d --build
```

如果本机有 reranker 模型，使用前一个仅 CPU 覆盖的命令即可；如果使用远程 reranker，填写 `ENTITY_RERANKER_URL` 和 `RERANKER_API_KEY`，并在应用配置中保持 `ENTITY_RERANKER_ENABLED=true`，不要使用 API-only 覆盖文件。

启动后检查：

```bash
curl http://127.0.0.1:8011/health
curl http://127.0.0.1:8080/graphs
curl http://127.0.0.1:8012/health
```

如果使用远程 reranker，最后一个地址替换为远程地址。应用页面默认在 `8011`，文件服务在 `8010`。

首次启动后进入“知识库”，创建项目并选择构建策略。模型配置是部署级提供方，模型名保存在项目中；改动提供方或模型后只影响后续构建，已有已发布图谱不会被自动改写。

## 9. systemd 用户服务

如果不用 Compose 应用容器，而是用 Python 虚拟环境启动：

```bash
cd /opt/knowledge-forge
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints.txt \
  -r requirements.txt -r requirements-docling.txt -r requirements-entity-resolution.txt
cp deploy/systemd/knowledge-forge.env.example ~/.config/knowledge-forge/environment
sed -i 's#^KNOWLEDGE_FORGE_HOME=.*#KNOWLEDGE_FORGE_HOME=/opt/knowledge-forge#' \
  ~/.config/knowledge-forge/environment
mkdir -p ~/.config/systemd/user
cp deploy/systemd/knowledge-forge.service ~/.config/systemd/user/
cp deploy/systemd/knowledge-forge-reranker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now knowledge-forge-reranker.service
systemctl --user enable --now knowledge-forge.service
```

两个 unit 不再假设用户名或仓库目录；路径由 `~/.config/knowledge-forge/environment` 中的 `KNOWLEDGE_FORGE_HOME` 决定。应用配置仍放在仓库根目录 `.env`，或者作为 systemd 环境变量提供。日志查看：

```bash
systemctl --user status knowledge-forge.service
journalctl --user -u knowledge-forge.service -f
journalctl --user -u knowledge-forge-reranker.service -f
```

若服务要在用户未登录时持续运行，启用 linger：

```bash
loginctl enable-linger "$USER"
```

## 10. 换机迁移清单

停机前备份：

1. PostgreSQL：项目、文档、结构块、知识块、向量、实体、关系、任务和会话；
2. `KnowledgeFiles/docs/`：源文档；
3. `KnowledgeFiles/chunks/`、`KnowledgeFiles/media/`：可复用的解析结果和图片证据；
4. `KnowledgeFiles/graphrag_native/`：已发布且仍需保留的 Native run bundle；
5. HugeGraph 数据卷或按 PostgreSQL 当前已发布快照重建；
6. `.env`/`.env.docker` 通过安全渠道重新配置，不复制到 Git；
7. reranker 模型缓存和 Docling/Hugging Face 缓存按需迁移。

迁移后先恢复 PostgreSQL，再恢复源文件和配置，启动 HugeGraph，确认 `knowledge_forge` 存在，最后启动应用。若 HugeGraph 丢失，可以依据 PostgreSQL 当前有效图谱快照重新物化；不要用旧的半成品目录直接覆盖新版本。

## 11. 常见故障

- **CPU Compose 仍提示 GPU 不可用**：确认命令同时包含根 Compose 和 `docker-compose.cpu.yml`，并运行 `docker compose config` 查看 `gpus` 和构建参数。
- **API 模型在页面没有选项**：填写 `*_API_MODELS`，模型名必须是远程服务真实接受的名称；点击“同步当前提供方模型”。
- **API 返回 404**：`*_API_BASE_URL` 应填服务的 `/v1` 根地址，不要填完整的 `/chat/completions` 或 `/embeddings`。
- **API 返回结果被拒绝**：检查 `choices[0].message.content`、`data[].embedding`、Embedding 向量维度和 API Key；Knowledge Forge 会拒绝空响应、截断响应和向量数量不一致。
- **GraphRAG Native 连不上模型**：确认 completion 与 embedding 两个 Base URL 都支持 OpenAI 兼容路径；Native 会先独立执行配置、连接和 Embedding 维度探针。
- **reranker 不健康**：查看容器日志和模型目录权限；短期可以关闭 `ENTITY_RERANKER_ENABLED`，不要把超时服务当成无知识结果。
- **HugeGraph 没有 `knowledge_forge`**：确认服务版本为 1.7.0、容器数据卷正确，再执行初始化脚本；不要清空数据卷。
- **页面访问不了**：检查 `SERVICE_HOST`、端口、防火墙和 `curl /health`；局域网绑定时还要按项目安全配置启用认证。

部署完成后，至少执行一次小文档的预处理、Knowledge Forge 构建、问答和引用展开，再执行一次 Native 配置/查询验收。这样可以验证的不只是进程存活，还包括模型、pgvector、HugeGraph、来源文件和 UI 入口的完整链路。
