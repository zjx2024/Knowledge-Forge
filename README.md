# Knowledge Forge

Knowledge Forge 是一个通用、多项目的知识库与知识图谱构建平台。它直接使用
PostgreSQL、HugeGraph 和 Ollama，不依赖 RAGFlow 运行时。

## 核心能力

- 支持 DOCX、PDF、Markdown、纯文本、XLSX、XLS 和 CSV 上传；解析器与分块策略按文档独立路由。
- 内置兼容、通用、论文和表格四种分块策略；通用策略保留章节父子结构，论文策略区分摘要、正文和参考文献，表格策略按行重复表头。
- 文档解析后可在页面逐项查看中间结构块，核对标题、正文、表格、页码、章节层级和标题路径。
- 构建过程分为结构解析、Chunk 分片与向量索引、实体关系与图谱构建三个阶段；阶段结果落库后，对应查看按钮立即解锁。
- 多文件上传会按提交顺序逐个处理，前一个文件完成全部三个阶段后再开始下一个文件，避免本地模型并发争用。
- PDF 默认先检查文本层与版面复杂度；普通文本层使用 PyPDF，扫描件或复杂版面自动回退到 Docling。
- 扫描 PDF、普通 PDF 内嵌图片和 DOCX 内嵌图片会提取到文件区，经 OCR 与项目视觉 LLM 生成带页码/章节来源的图片知识；这些内容与正文共同参与分块、向量索引、实体关系抽取、跨文档归一和图谱合并。独立 PNG/JPG 暂不作为上传文档类型。
- 使用内容 SHA-256 在项目内去重，相同文档可进入不同项目。
- 使用 Ollama Embedding 和 pgvector HNSW 建立向量索引，联合向量相似度和关键词分数检索。
- 每个项目从通用逻辑 Schema 起步；系统从正文中分层选取语义多样的代表性知识块，LLM 根据证据提出追加类型，系统校验后生成新版本。
- Schema 建议失败时保留当前版本；用户可手动触发全项目 Schema 重新评估，删除最后一篇文档后自动恢复通用基线 v1。
- 抽取实体提及和关系断言，并保留项目、文档、章节、知识块、原文证据和字符位置来源。
- 实体关系抽取采用证据约束主抽取；可疑知识块执行一次针对性补抽取，不确定关系再经过规则校验和一次批量 LLM 复核。
- 通过名称规范化、别名和向量相似度保守合并多文档实体；不确定时不合并。
- 使用独立 HugeGraph 物理图 `knowledge_forge`，多个项目通过 `project_id/project_key` 隔离。
- 任务状态持久化到 PostgreSQL，页面切换或服务重启不会丢失任务记录。
- 任务面板分别展示结构解析、分块及向量索引、Schema 和基础图数据、逐块抽取实体与关系、HugeGraph 同步五个子任务的独立进度。
- 文档删除会先立即软删除并取消关联任务，使其马上从项目列表和检索结果中移除；HugeGraph、源文件和过程文件由持久化后台清理任务处理，服务重启后可继续恢复。清理完成后会再次停用该文档的知识块、向量、实体提及和关系贡献，不影响其他文档仍支持的实体。

## 架构与目录

```text
knowledge-forge/
├── knowledge_forge/             # 应用模块
├── KnowledgeFiles/
│   ├── docs/<项目目录>/          # 源文档
│   ├── chunks/<项目目录>/        # 章节和知识块过程 JSON
│   ├── media/<项目目录>/<文档>/   # PDF/DOCX 内嵌图片及视觉分析缓存
│   └── .trash/                   # 软删除文件
├── deploy/hugegraph/             # 独立物理图配置
├── docs/IMPLEMENTATION_PLAN.md   # 架构基线与实施阶段
├── tests/
├── AGENTS.md                     # 持续开发约束
├── .env.example
├── .env.docker.example
└── docker-compose.yml
```

运行时职责：

- PostgreSQL + pgvector：元数据、章节、知识块、HNSW 向量索引、Schema 版本、实体、提及、关系断言和任务。
- HugeGraph：可遍历的文档、知识块、规范实体和实体关系。
- Ollama：模型发现、向量生成、逻辑 Schema 评估和结构化图谱抽取。

## 前置条件

1. PostgreSQL 16 + pgvector 可访问；Docker 部署会基于 `postgres:16` 自动构建并启动对应数据库。
2. Ollama 可访问，至少有一个 LLM 和一个 Embedding 模型。
3. HugeGraph 1.7 可访问。
4. Python 3.12 或 Docker Compose。

Docling 用于处理扫描 PDF 或复杂论文版面，默认随项目依赖安装：

```bash
.venv/bin/python -m pip install -c constraints.txt -r requirements-docling.txt
```

Docling 首次处理复杂文档时会下载版面分析和 OCR 模型。Docker Compose 将
`/root/.cache` 挂载到 `knowledge-forge-docling-cache` 卷，容器重建后可继续复用。
默认使用 `HF_ENDPOINT=https://hf-mirror.com` 并关闭 Xet 下载通道；可在环境文件中覆盖。
Docker 默认向 Docling 暴露 NVIDIA GPU。PDF 版面分析、表格识别和 OCR 使用 CUDA，
批大小默认为 1；开始前会检查可用显存，并将本进程显存上限控制为 GPU 总量的 20%。
可用显存低于 6144 MiB 或遇到 CUDA OOM 时，会释放 CUDA 缓存并自动用 CPU 重试该文档。
Word、纯文本和电子表格的原生结构读取属于 I/O 与规则解析，不包含可迁移到 GPU 的模型计算。

```dotenv
DOCLING_DEVICE=auto
DOCLING_CUDA_FALLBACK=true
DOCLING_GPU_MIN_FREE_MB=6144
DOCLING_GPU_MEMORY_FRACTION=0.20
DOCLING_BATCH_SIZE=1
DOCLING_NUM_THREADS=4
```

默认模型为：

```text
LLM: qwen3.6:latest
Embedding: bge-m3:latest
```

可在新建项目或项目设置中选择 Ollama 已安装的其他模型。项目设置同时保存“项目上下文
Token 上限”和“单切片 Token 上限”：前者是应用侧请求预算，不会向 Ollama 发送
`num_ctx`，也不会覆盖容器的 `OLLAMA_CONTEXT_LENGTH`；后者约束该项目后续文档生成的
每个知识块。页面会根据所选 LLM 和 Embedding 模型动态限制可填最大值。

Ollama 请求使用显式、互不混用的预算。所有数值单位均为 token：

```dotenv
# runner 尚未加载或 /api/ps 不可用时的保守有效上下文；必须大于 0
OLLAMA_EFFECTIVE_CONTEXT_FALLBACK_TOKENS=8192
# 未提供项目预算时的应用侧总上下文预算；不会映射为 options.num_ctx
OLLAMA_DEFAULT_CONTEXT_BUDGET_TOKENS=8192
# 默认输出上限，精确映射为 options.num_predict
OLLAMA_DEFAULT_MAX_OUTPUT_TOKENS=2048
# Prompt 模板、JSON Schema 和估算误差预留；允许 0，建议至少 128
OLLAMA_SAFETY_MARGIN_TOKENS=256
# 估算 token 少于实际 prompt_eval_count 时允许的绝对和相对误差
OLLAMA_TOKEN_VARIANCE_ABSOLUTE_TOKENS=128
OLLAMA_TOKEN_VARIANCE_RATIO=0.50
# /api/show 与 /api/ps 能力结果缓存秒数；0 表示每次重新读取
OLLAMA_CONTEXT_CACHE_SECONDS=60
# 每张输入图片的保守 token 估算
OLLAMA_IMAGE_TOKEN_ESTIMATE=1024
```

应用通过 `/api/show` 读取模型声明上限，通过 `/api/ps` 读取已加载 runner 的实际
`context_length`，请求准入按两者和项目预算的最小值判断。准入条件为“估算输入 +
最大输出 + 安全预留不超过有效上下文”。响应若为 `done_reason=length`、未完成，或
`prompt_eval_count` 明显超过估算值，会失败关闭，不使用可能被截断的 JSON。

内嵌图片默认开启 OCR 和视觉理解；可通过以下环境变量调整尺寸过滤与扫描页渲染。视觉
请求使用项目选定的 LLM，并沿用项目上下文预算。

```dotenv
IMAGE_ANALYSIS_ENABLED=true
IMAGE_OCR_ENABLED=true
IMAGE_MIN_WIDTH=96
IMAGE_MIN_HEIGHT=96
IMAGE_MIN_PIXELS=16000
IMAGE_MAX_DIMENSION=2048
IMAGE_SCAN_TEXT_THRESHOLD=40
IMAGE_PDF_RENDER_SCALE=1.7
```

## 初始化独立 HugeGraph 图

本项目不写入已有的 `hugegraph` 或 `test`，而是使用物理图
`knowledge_forge`。一台 HugeGraph 服务只需初始化一次，之后所有业务项目共享
这张物理图并通过项目字段隔离。

如果 HugeGraph 运行在名为 `hugegraph` 的 Docker 容器中：

```bash
cd /home/kdfy/zjx-projects/knowledge-forge
sudo ./deploy/init_hugegraph_graph.sh hugegraph hugegraph/hugegraph:latest
curl http://127.0.0.1:8080/graphs
```

正常结果应包含：

```json
{"graphs":["hugegraph","test","knowledge_forge"]}
```

脚本会复制 `deploy/hugegraph/knowledge_forge.properties`、停止容器、执行
HugeGraph 官方 `init-store.sh`，然后重新启动容器。不会清空其他物理图。

## 本地开发启动

```bash
cd /home/kdfy/zjx-projects/knowledge-forge
cp -n .env.example .env
python3.12 -m venv .venv
.venv/bin/python -m pip install pip==25.3
.venv/bin/python -m pip install -c constraints.txt \
  -r requirements.txt -r requirements-docling.txt -r requirements-entity-resolution.txt
.venv/bin/python server.py
```

`python3.12` 必须是独立 CPython 3.12，不能指向 Conda 环境。项目的正式本地开发
入口始终是根目录 `.venv`；已有 Conda 环境只可作为迁移回退或依赖对照，不参与日常启动。
VS Code 中使用“Python: Select Interpreter”并选择
`<仓库根目录>/.venv/bin/python`。

依赖中固定包含 `graphrag==3.1.1`，用于 Python 3.12 运行兼容和后续评估；当前业务管线
尚未接入 GraphRAG BYOG，Knowledge Forge 仍使用自身的实体、关系和 HugeGraph 写入流程。

### 依赖锁更新

`constraints.txt` 与 `constraints-gpu.txt` 是 Python 3.12/Linux x86_64 的完整
传递依赖锁；CPU 与 CUDA 文件只在 PyTorch wheel 变体上不同。更新任一
`requirements*.txt` 后，在同一个新环境中重新解析并生成锁：

```bash
.venv/bin/python -m pip install --dry-run \
  -c constraints.txt --report /tmp/knowledge-forge-py312-resolver-report.json \
  -r requirements.txt -r requirements-docling.txt -r requirements-entity-resolution.txt
.venv/bin/python scripts/generate_python312_constraints.py \
  --report /tmp/knowledge-forge-py312-resolver-report.json
```

之后必须按锁文件重新安装，并执行 `pip check`、测试、PostgreSQL 和 Docker 门禁。

访问：

```text
管理页面：http://<SERVICE_HOST>:8011
文件服务：http://<SERVICE_HOST>:8010
Hubble 图管理：http://<SERVICE_HOST>:18088
```

本地开发可先单独启动 Compose 中的 pgvector 数据库：

```bash
cp -n .env.docker.example .env.docker
sudo docker compose up -d postgres
```

应用启动时会创建 `POSTGRES_DB=knowledge_forge`、`vector`/`pg_trgm` 扩展和全部业务表，
并将历史数组向量自动回填到 pgvector。PostgreSQL 用户需要具有创建扩展和表的权限。

## Docker 启动

```bash
cp -n .env.docker.example .env.docker
# 修改 SERVICE_HOST、HTTP_BASE_URL、数据库密码等实际配置
sudo docker compose up -d --build
sudo docker compose logs -f knowledge-forge
```

停止：

```bash
sudo docker compose down
```

Compose 使用独立的 `knowledge-forge-postgres` 保存业务数据，通过
`host.docker.internal` 访问宿主机上的 Ollama 和 HugeGraph；浏览器访问地址仍由
`SERVICE_HOST/HTTP_BASE_URL` 决定。宿主机可通过 `5433` 端口访问该数据库。

## 文档构建流程

1. 创建项目并选择抽取 LLM、Embedding 模型。
2. 上传源文档。默认由系统自动选择解析器和分块策略，也可在“解析与分块设置”中手动指定。
3. 点击“上传解析”。后台按以下阶段执行，并在日志中分别显示：
   - 阶段一：解析文档结构；完成后可点击文档行中的“结构块”。
     PDF/DOCX 内嵌图片在本阶段完成提取、OCR 和视觉理解，并形成 `image` 结构块。
   - 阶段二：生成 Chunk 分片并建立向量索引；完成后可点击“Chunk 分片”。
   - 阶段三：评估逻辑 Schema、抽取并归一实体/关系、同步 HugeGraph；完成后可点击“查看知识图谱”。
   批量上传时，每个文件完整走完三个阶段后，才顺序处理下一个文件。
4. 在“联合检索”中查询知识块、实体和关系，并查看来源文档链接。
5. 文档入库完成后会自动同步 HugeGraph；点击“查看知识图谱”会进入 Hubble Gremlin
   分析页，填入当前项目的只读查询并自动执行。结果保留项目全图，并额外展示
   “源文档 -> 切片 -> 跨文档共享实体”来源路径。该能力使用 `deploy/hubble` 中的定制 Hubble 镜像。

处理配置按文档保存：`requested_parser/requested_chunk_strategy` 记录用户请求，
`effective_parser/effective_chunk_strategy` 记录实际执行结果，同时保存解析器版本、
质量报告、路由原因和内容版本。手动选择与文件类型不兼容时会在保存文件前拒绝；
重新解析仍只替换该文档的章节、知识块、向量和图谱贡献。

## Schema 策略

HugeGraph 物理 Schema 固定为：

```text
Vertex: KFDocument, KFChunk, KFEntity
Edge:   kf_contains, kf_mentions, kf_related
```

领域差异保存在 PostgreSQL `graph_schemas.schema_json` 中。每个项目都有独立、
可追溯的逻辑 Schema 版本。Schema 评估不会固定读取文档开头：系统先过滤目录、
封面和过短知识块，再按章节分层，并利用已生成的 Embedding 贪心选择语义多样的
代表性知识块。代表块分批交给 LLM 发现候选类型，随后进行全局合并；候选必须引用
真实知识块证据，并通过置信度、证据多样性、重复类型和单次新增数量校验。

已有项目文档可覆盖新文档内容时，系统根据跨文档向量新颖度跳过 Schema LLM 评审，
减少重复调用。自动演进始终只允许追加实体类型和关系类型，不会自动删除、重命名或
改变已有类型含义。代表块数量会随有效知识块数和章节数做次线性增长，并受块数、总字符数
和单块字符数三重上限保护。采样基线与上限、新颖度阈值和证据门槛可通过
`SCHEMA_SAMPLE_*`、`SCHEMA_REVIEW_*`、`SCHEMA_MAX_*_ADDITIONS` 配置。

项目统计区的“重新评估 Schema”会明确地从基线 v1 出发，使用当前全部有效且已入库
文档重新生成活动版本，但不会自动重跑既有实体和关系抽取。项目不再包含有效文档时，
系统会自动激活基线 v1。

## 实体与关系抽取

每个实体和关系必须携带能够在完整原知识块中精确定位的证据文本。实体名称及参与验证的
别名必须在正文中具有真实来源 span；短 ASCII 名称使用词边界匹配。关系端点、谓词和方向
由严格 verifier 按知识块批量复核，单纯共现、类型兼容或模型自报高分不会自动发布。
默认搜索、关系列表、首页统计、关系 ID 回取和 HugeGraph 同步统一只消费人工审核的
`manual_verified`，以及携带当前验证策略版本的 `rules_verified` / `llm_verified`。
其他状态和历史策略关系保留用于审计，但不会进入产品查询、关系向量召回或 HugeGraph。

信息量较高但主抽取为空，或已经抽取多个实体却遗漏明显关系的知识块，会额外执行一次
针对性补抽取。相关行为可通过 `EXTRACTION_GLEANING_*`、
`EXTRACTION_VERIFICATION_ENABLED` 和 `EXTRACTION_RULE_ACCEPT_CONFIDENCE` 调整；
任务日志会显示补抽取、复核通过和拒绝数量。

## 实体归一策略

实体归一按项目隔离，并使用分层候选召回：先按 Unicode 规范名称和规范别名精确匹配；
未命中时，分别使用 PostgreSQL `pg_trgm` 召回名称及别名候选、使用 pgvector HNSW
召回身份语义候选。查询实体和历史实体使用相同的“名称、类型、别名、描述”向量文本，
不把整段 Chunk 上下文混入身份向量。两路候选按实体 ID 合并后，计算字符形态、词形、
包含关系、缩写、类型、编号/版本冲突和向量分数，再由本地 `bge-reranker-v2-m3`
交叉编码器重排少量候选。缩写/全称只是名称形态特征之一，不是专用合并规则。
名称、向量和重排候选数量会根据项目中同类型实体规模按对数级增长，并分别受
`ENTITY_LEXICAL_MAX_K`、`ENTITY_VECTOR_MAX_K` 和 `ENTITY_RERANKER_MAX_K` 保护。

重排分数表示语义相关度，不直接等同于实体身份概率。自动合并必须同时满足强名称证据、
类型兼容、无编号冲突和综合阈值；语义相关但身份不确定的候选交给项目 LLM 做最终裁决，
无可靠候选时创建新实体。这样可覆盖人物、组织、产品、地点、标准、事件等不同领域，
同时避免把上下级组织、同类产品或地理包含关系误合并。

每次决策都会写入 `entity_resolution_decisions`，保存候选实体、各项分数、决策方式、
模型和原因。LLM 调用失败或结果不满足接受阈值时默认保留为独立实体，不会冒险合并。
Top-K、召回阈值、重排器路径和裁决阈值可通过 `ENTITY_*` 环境变量调整。重排器使用
Transformers 在项目容器内离线推理；Ollama 当前没有 `/api/rerank`，仍只负责 LLM 与
Embedding。默认模型目录为宿主机 `/home/kdfy/models/bge-reranker-v2-m3`，Compose
以只读方式挂载，模型缺失、加载失败或 GPU OOM 时自动回退到规则、向量和 LLM 链路。

## 向量存储说明

向量以 pgvector `vector` 保存，并按 Embedding 模型和向量维度自动创建余弦距离
HNSW 索引。查询先在 PostgreSQL 中执行项目、对象类型、模型和维度过滤及向量召回，
再结合现有中文关键词评分重排；包含完整查询词的切片会作为关键词候选一并参与排序。

`DOUBLE PRECISION[]` 兼容列暂时保留，用于无损迁移和回滚；实体归一直接通过 pgvector
Top-K 检索候选，不再将项目全部实体向量加载到 Python 中逐个比较。
应用启动会自动迁移旧数据，不会删除已有向量。超过 2000 维的模型仍使用 pgvector
精确检索，不创建 HNSW `vector` 索引。

## 主要数据表

- `projects`、`documents`、`sections`、`chunks`
- `vector_embeddings`
- `graph_schemas`
- `canonical_entities`
- `entity_mentions`
- `entity_resolution_decisions`
- `relation_assertions`
- `relation_candidates`、`relation_verification_events`
- `jobs`、`models`、`settings`

Knowledge Forge 不依赖 RAGFlow；项目、文档、知识块和向量均由本项目自己的 PostgreSQL 表管理。

## P0 verifier 策略升级

当前关系验证策略版本为 `relation-verifier-p0-v1`，图发布策略版本为
`relation-graph-p0-v1`。升级后首次启动会幂等执行数据库迁移：缺少当前策略版本的历史
`rules_verified` 和 `llm_verified` 会降级为 `legacy_unverified`；非发布关系及没有有效断言
的孤儿 relation vector 会被停用。关系候选、验证事件和断言组成单个 PostgreSQL
“关系审计链事务”，这不代表实体、向量和外部图写入具备跨系统事务。

关系断言是权威数据，关系向量是可重建索引。Embedding 写入失败时，系统会写入
`indexing_failures` 补偿记录，不会降级已经验证的断言；后续项目图重建会重试这些索引。
数据库迁移不会在启动时静默执行耗时的 HugeGraph 全量重建。

服务升级并启动后，必须为每个已有项目显式触发一次图谱重同步：

```bash
curl -X POST http://127.0.0.1:8011/api/projects/<PROJECT_ID>/graph
```

已有非空图在策略版本不匹配时会标记为 `stale`。重建任务依次进入 `rebuilding`，先清除
项目在 HugeGraph 中的旧贡献，再从 PostgreSQL 按当前发布谓词完整重建，并比较
PostgreSQL 发布关系数、本次写入数和 HugeGraph 实际关系数。只有三者一致才进入
`ready` 并写入当前图策略版本；任一阶段失败均进入 `failed`。`stale`、`rebuilding` 和
`failed` 状态下，服务端会拒绝生成 Hubble URL，直接访问 `/graph/view` 返回 HTTP 409。

## P1-P3 策略、性能与回滚

### 图代次与任务合并

项目知识发生变化时，PostgreSQL 会递增 `graph_generation` 并把图标记为 `stale`。
入库任务繁忙期间，后续请求只更新同一个待执行图任务的目标代次；系统在项目进入静默点后
执行一次完整重建，因此连续入库 10 篇文档通常从 10 次全图重建降为 1 次。重建期间又产生
新代次时，旧代次不能进入 `ready`，而会保留 pending 状态并由新代次接管。图清理、逐文档
写入、关系数量核对或最终状态更新任一步失败，项目都会保持 `failed/stale`，可重试但不会
暴露部分图。

`relation_assertions` 是不可变来源事实，当前图端点由可重建的关系投影决定。实体合并、拆分、
撤销或项目策略升级都会刷新投影、递增图代次并要求重建，原始 mention 和 assertion 不会被
覆盖。

### 策略版本与启用门禁

构建策略按实体抽取、实体归一、关系候选、关系分类、关系验证和 Schema 六个版本轴组成完整
快照。当前生产路径继续使用 `control`；P1 关系策略与 P2 实体策略仅以隔离的 `shadow`
namespace 运行，不写正式实体、断言、向量、搜索结果或 HugeGraph。发布顺序为
`control -> shadow -> canary -> limited -> expanded`，每一步均记录前一快照和回滚快照。

非 control 快照只有在以下条件全部满足后才能激活：快照覆盖全部 indexed 文档、冻结数据集
manifest 有效、标签由人工确认、holdout 与 dev 隔离、质量和成本门禁通过，并且数据库中
存在 `production_approved` gate。仓库内 `tests/fixtures/kg_gold` 仅用于工程验证，明确不是
人工 Gold，不能授权生产启用。因此当前新策略状态为 `engineering_complete`，默认仍关闭。

### Prompt、批处理与溢出

关系分类先生成有稳定 ID 的 mention/candidate，再按项目上下文预算分批。每批 Prompt 中
Chunk 正文只出现一次，并为输出和安全余量预留非零 Token；单个候选超过预算时会显式进入
`deferred`，不会使用切片截断或静默丢弃。模型必须为每个 admitted candidate 返回唯一
决策，缺失、重复、非法方向或越出 Schema 的谓词均 fail-closed。相关上限由
`RELATION_CLASSIFIER_OUTPUT_RESERVE`、`RELATION_CLASSIFIER_SAFETY_MARGIN`、
`RELATION_CLASSIFIER_MAX_CANDIDATES` 和项目上下文 Token 上限共同控制。

实体 exact 候选按 Chunk 批量预取，命中后不调用 Embedding、reranker 或 LLM；未命中的
身份文本按 `ENTITY_EMBEDDING_BATCH_SIZE` 批量向量化。reranker 只处理有界 Top-K，所有
外部调用、候选数、Token、数据库语句、重试和图重建均记录操作数指标，但不记录正文、完整
Prompt、模型完整响应或凭据。

### 数据库迁移与 HNSW

应用启动时幂等创建规范 alias 表、实体到来源文档映射、mention assignment、cannot-link、
shadow、策略快照、质量门禁、rollout、图代次和审计表；旧实体的规范名/别名及来源计数可
重复回填，不删除兼容列。历史项目应逐项目显式触发策略快照和图重建，不在启动时做破坏性
全量重入库。

pgvector 会按 Embedding 模型和维度创建 partial HNSW 索引。维度不超过
`PGVECTOR_HNSW_MAX_DIMENSIONS` 时必须走 HNSW Top-K；索引创建失败不会写入成功缓存，
下一次会重试。超维模型进入可观测 degraded 精确扫描模式。真实 PostgreSQL 的 exact、
trigram、来源映射和 HNSW 查询计划可用以下命令验证：

```bash
.venv/bin/python scripts/run_disposable_postgres_tests.py tests/test_p1_postgres_plans.py
```

脚本只在临时数据库生成 10k/100k 验证数据，测试结束无论成功或失败都会终止连接并删除该
数据库，不会污染业务库。1M 基准只在独立资源充足的环境显式设置
`RUN_LARGE_BENCHMARKS=1` 后运行。

### 实体修订与回滚

人工或受门禁策略可以追加实体 merge、mention split 和 undo 事件。每次操作保留操作者、
原因、分数、证据、操作前后 assignment 和 redirect；merge 不重写原始关系断言，split
只移动选定 mention，undo 通过补偿事件恢复旧 assignment。修改完成后关系投影、来源计数、
实体向量和 HugeGraph 端点均按新代次重建。cannot-link 优先于 exact/alias 规则，存在冲突
时禁止自动合并。

### 质量评估与外部 smoke

工程 fixture、操作数基准和人工标注包命令：

```bash
.venv/bin/python benchmarks/benchmark_kg_construction.py
.venv/bin/python benchmarks/build_annotation_bundle.py --help
.venv/bin/python benchmarks/evaluate_kg_gold.py --help
```

质量报告覆盖实体 exact/overlap span、类型 macro-F1、候选召回、pairwise/B3 聚类、严格关系
与证据、Brier/ECE 校准、control/treatment discordant cases 和 Wilson 区间。生产报告必须
使用独立人工确认且冻结的 Gold/holdout manifest；模型自产标签只能形成待人工审核队列。

默认 pytest 不访问外部服务。环境具备时显式运行：

```bash
RUN_LIVE_MODEL_TESTS=1 .venv/bin/python -m pytest -q -m live_model
RUN_HUGEGRAPH_INTEGRATION=1 .venv/bin/python -m pytest -q -m hugegraph_integration
```

模型 smoke 验证 Ollama 模型 digest/context、严格 JSON、Embedding 数量/顺序/维度/finite 和
本地 reranker pair/score 对齐。HugeGraph smoke 只做只读图发现；purge、失败注入、409、
重建和 retry 必须使用明确隔离物理图，不能在承载用户数据的 `knowledge_forge` 图中执行。
fake HugeGraph 测试覆盖进程状态恢复、代次变化、数量不一致、失败重试和发布前 409。

## 验证

上线前 revision、shadow snapshot、Gold/release gate、HugeGraph sentinel、迁移恢复和隔离
E2E 的完整操作见 [`docs/pre-live-readiness.md`](docs/pre-live-readiness.md)。

```bash
.venv/bin/python -m py_compile knowledge_forge/*.py server.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_disposable_postgres_tests.py
curl http://127.0.0.1:8011/health
curl http://127.0.0.1:8080/graphs
```

当前自动化测试的通过数量以本机执行结果为准。Docker 首次构建需要本机已有
`python:3.12-slim`，或能够访问镜像仓库下载该基础镜像。
