# Knowledge Forge

Knowledge Forge 是一个通用、多项目的知识库、知识图谱与证据约束问答平台。它直接使用
PostgreSQL、HugeGraph、Ollama 和独立 reranker 服务，不依赖 RAGFlow 运行时。

## 核心能力

- 支持 DOCX、PDF、Markdown、纯文本、XLSX、XLS 和 CSV 上传；解析器与分块策略按文档独立路由。
- 内置兼容、通用、论文和表格四种分块策略；通用策略保留章节父子结构，并在章节过密时把连续微型子节合并为保留全部内部标题和来源键的有界结构块（`GENERAL_COALESCE_MIN_SECTIONS`，设为 `0` 可关闭），论文策略区分摘要、正文和参考文献，表格策略按行重复表头。对正文主语言明确、标题成对且篇幅相近的双语摘要/概述，主语言版本参与图谱抽取，次语言镜像版本保留向量检索但不重复生成图事实；OCR 把译文标题、作者、`Abstract` 或摘要正文拆到相邻节/下一页时，也按页码与文字脚本连续性恢复完整镜像前置内容。普通双语正文与未成对摘要不受影响。参考文献支持全角编号和跨页续排，作者简介作为可检索的文后资料但不生成正文图事实；论文末页若因双栏顺序混入下一篇文章预览，只有在新摘要、关键词、DOI 与旧参考文献续接同时出现时才隔离该预览。
- 文档解析后可在页面逐项查看中间结构块与知识块，核对标题、正文、页码、章节层级和标题路径；块中关联的图片会显示缩略图，表格会显示行列预览，均可点击打开完整内容。
- 构建过程分为结构解析、Chunk 分片与向量索引、实体关系与图谱构建三个阶段；阶段结果落库后，对应查看按钮立即解锁。
- 多文件上传会按提交顺序逐个处理，前一个文件完成全部三个阶段后再开始下一个文件，避免本地模型并发争用。
- PDF 默认先检查文本层与版面复杂度；普通文本层使用 PyPDF，扫描件或复杂版面自动回退到 Docling。
- 扫描 PDF、普通 PDF 内嵌图片和 DOCX 内嵌图片会提取到文件区，经 OCR 与项目视觉 LLM 生成带页码/章节来源的图片知识；这些内容与正文共同参与分块、向量索引、实体关系抽取、跨文档归一和图谱合并。独立 PNG/JPG 暂不作为上传文档类型。
- 使用内容 SHA-256 在项目内去重，相同文档可进入不同项目。
- 使用 Ollama Embedding 和 pgvector HNSW 建立向量索引，联合向量相似度和关键词分数检索。
- 联合检索从查询表面词和实体语义召回生成少量图种子，执行有界一至两跳扩展；关系按语义、来源证据和网络距离综合排序，并优先保留不同实体对，避免同一对的重复证据挤占 Top-K。
- Knowledge Forge 图谱发布后异步生成证据约束的详细社区报告；报告未完成时检索继续使用知识块、实体和关系，已完成报告作为宏观主题检索增强，并保留所引用关系的文件、页码和原文证据。
- 每个项目从通用逻辑 Schema 起步；普通构建冻结并复用当前活动版本，先发布有精确原文证据的事实，不让 Schema 缺项成为召回门槛。
- 用户可通过独立的全项目 Schema 重新评估任务，从代表性正文证据归纳追加类型；失败时保留当前版本，删除最后一篇文档后自动恢复通用基线 v1。
- 抽取实体提及和关系断言，并保留项目、文档、章节、知识块、原文证据和字符位置来源。
- 实体发布前执行与格式无关的边界审计：带情态词、完整主谓/分类/被动结构、续接连词或跨软换行未闭合的整句不会因模型误标成事件、组织或技术而成为图节点；明确的事件名、名词化方法、列表名词项、表格单元格和关系端点仍按来源证据保留。
- 实体关系抽取采用证据约束主抽取；可疑知识块执行一次针对性补抽取，不确定关系再经过规则校验和一次批量 LLM 复核。
- 通过名称规范化、别名和向量相似度保守合并多文档实体；不确定时不合并。
- 使用独立 HugeGraph 物理图 `knowledge_forge`，多个项目通过 `project_id/project_key` 隔离。
- 任务状态持久化到 PostgreSQL，页面切换或服务重启不会丢失任务记录。
- 任务面板清晰区分总进度和当前步骤进度；每一步将状态与处理说明分开显示，避免长说明挤压布局。
- 文档删除会先立即软删除并取消关联任务，使其马上从项目列表和检索结果中移除；HugeGraph、源文件和过程文件由持久化后台清理任务处理，服务重启后可继续恢复。清理完成后会再次停用该文档的知识块、向量、实体提及和关系贡献，不影响其他文档仍支持的实体。
- 提供 Knowledge Forge 与 GraphRAG Native 两种项目级构建策略；两者共享源文档与预处理结果，但实体、关系、社区和发布状态互相隔离。
- 提供显式单知识库智能问答：Knowledge Forge 融合当前知识块、实体、关系和已完成社区报告，并在多社区或关系稀疏问题上使用有界社区联系导航补召回；GraphRAG Native 只读取当前已验证的不可变 Bundle。通用检索计划按事实、实体、关系、过程、比较、全局、来源和图表问题动态分配文本/实体/关系/社区候选，并在最终上下文中按实际槽位缩放配额、自动回填。候选经 RRF 与内部 reranker 重排后，只把来源完整的证据交给一次回答生成，并由服务器逐项核验引用。多轮追问先由 LangGraph 语境子图补全为独立检索问题；无法唯一解析的指代会先请求澄清，不会带着猜测进入检索。

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
├── docs/PROJECT_STATUS.md        # 当前产品能力、边界与下一阶段
├── docs/knowledge-agent-implementation-roadmap.md
│                                # 智能体架构、实现状态与阶段路线
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
- Reranker HTTP 服务：独立加载交叉编码器，为实体归一候选和智能体检索候选提供有界相关度重排。

## 前置条件

1. PostgreSQL 16 + pgvector 可访问；Docker 部署会基于 `postgres:16` 自动构建并启动对应数据库。
2. Ollama 可访问，至少有一个 LLM 和一个 Embedding 模型。
3. HugeGraph 1.7 可访问。
4. `bge-reranker-v2-m3` 本地模型目录可读；不启用实体重排时可明确关闭。
5. Python 3.12 或 Docker Compose。

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

`OLLAMA_BASE_URL` 用于模型发现和结构化生成。需要把生成与向量计算放在两个已独立验证
的 Ollama 实例时，可设置 `OLLAMA_EMBEDDING_BASE_URL`；未设置时它严格回退到
`OLLAMA_BASE_URL`，现有单实例部署行为不变。`OLLAMA_EMBEDDING_RUNTIME_PROFILE`
标识向量数值的执行环境；更换 CPU/GPU 或底层实现时应更新该值。系统会保留原有结构解析
与知识块，只重建不匹配的向量索引。同一 Embedding 模型和抽取策略生成的实体事实快照
仍可续用，不会仅因 CPU/GPU 切换而重做耗时的 LLM 抽取；新写入的实体向量会记录实际
执行环境以便审计。

已有项目可在切换后执行可中断、可续跑的向量迁移；它只更新当前有效的 pgvector 索引：

```bash
.venv/bin/python scripts/reindex_embedding_runtime.py --dry-run
.venv/bin/python scripts/reindex_embedding_runtime.py
```

Ollama 请求使用显式、互不混用的预算。所有数值单位均为 token：

```dotenv
# runner 尚未加载或 /api/ps 不可用时的保守有效上下文；必须大于 0
OLLAMA_EFFECTIVE_CONTEXT_FALLBACK_TOKENS=8192
# 固定模型生成种子，使同一输入在重试和服务重启后尽量得到可比较结果
OLLAMA_SEED=17
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

这里的 `8080` 是 Docker 示例端口。已有部署应始终使用 `.env` 中的
`HUGEGRAPH_HOST/HUGEGRAPH_PORT`；本仓库当前本机部署使用 `127.0.0.1:8081`，不能把示例端口
当成运行实例的固定地址。

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

依赖中固定包含 `graphrag==3.1.1`。界面提供 `Knowledge Forge` 与
`GraphRAG Native` 两种知识构建策略。GraphRAG Native 从当前 active revision 的已持久化结构块
生成隔离的 GraphRAG Standard 输入，产物保存在
`KnowledgeFiles/graphrag_native/` 的不可变 run bundle 中。该策略只支持项目级全量
构建，不写入旧实体、关系、pgvector 或 HugeGraph；其应用内查看器、社区报告和查询
也不依赖 Hubble。文档新增、删除或重建 revision 会使 native run 过期，必须重新构建。
native run 内部的 READY 指针不改变项目当前策略选择，也不会自动发布或部署。
Native Standard 工作流的隔离子进程预算由 `GRAPHRAG_NATIVE_TIMEOUT_SECONDS` 控制，默认
为 7200 秒；超时只会使该 run 失败，阶段性 staging 产物不会被标记为可查询成果。

真实模型构建与查询会消耗本地 Ollama 资源，因此普通测试只运行离线 bundle、配置、
迁移和路由门禁；需要人工授权后才运行真实 GraphRAG index/query。离线验证工具：

```bash
.venv/bin/python scripts/validate_graphrag_native_bundle.py --root '<RUN_DIR>'
.venv/bin/python scripts/check_graphrag_native_freshness.py --project-id '<PROJECT_ID>' --run-id '<RUN_ID>'
.venv/bin/python scripts/retention_graphrag_native.py --project-id '<PROJECT_ID>'
```

retention 命令默认只列出候选；即使显式执行清理，也会拒绝 active、最后一个 READY
或仍持有 lease 的 run。

完整的数据边界、状态机、API 与恢复说明见
[GraphRAG Native 策略](docs/graphrag-native.md)。

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
     PDF/DOCX 内嵌图片在本阶段完成提取、OCR 和视觉理解，并形成 `image` 结构块；图片和表格可在结构块列表中直接预览并打开完整内容。
   - 阶段二：生成 Chunk 分片并建立向量索引；完成后可点击“知识块”。知识块会继续携带关联图片、表格和页码来源，并提供相同的预览与放大查看能力。
   - 阶段三：冻结活动逻辑 Schema、抽取并归一实体/关系、同步 HugeGraph；完成后可点击“查看知识图谱”。
   批量上传时，每个文件完整走完三个阶段后，才顺序处理下一个文件。
4. 在“联合检索”中查询知识块、实体和关系，并查看来源文档链接。
5. 文档入库完成后会自动同步 HugeGraph；点击“查看知识图谱”会进入 Hubble Gremlin
   分析页，填入当前项目的只读查询并自动执行。结果保留项目全图，并额外展示
   “源文档 -> 切片 -> 跨文档共享实体”来源路径。该能力使用 `deploy/hubble` 中的定制 Hubble 镜像。

处理配置按文档保存：`requested_parser/requested_chunk_strategy` 记录用户请求，
`effective_parser/effective_chunk_strategy` 记录实际执行结果，同时保存解析器版本、
质量报告、路由原因和内容版本。手动选择与文件类型不兼容时会在保存文件前拒绝；
重新解析仍只替换该文档的章节、知识块、向量和图谱贡献。

### Knowledge Forge 详细社区报告

社区检测和图谱发布是前置的确定性阶段，详细报告是随后运行的异步增强任务，不会延迟图谱
可用状态。至少包含 2 个实体和 1 条已验证内部关系的社区具备报告资格；系统按层级、实体数
和内部关系数排序，初次只自动生成前 `KF_COMMUNITY_REPORT_INITIAL_LIMIT` 个，避免长文档或
多文档项目一次排队数百份模型报告。其余事实充分的社区保留为按需任务，用户打开该社区时
生成一份，检索命中其中实体时每次最多激活
`KF_COMMUNITY_REPORT_SEARCH_ACTIVATION_LIMIT` 份；事实不足的社区明确标为跳过。

每份报告只接收该社区中的规范实体、已验证内部关系和对应原文证据。代表性关系不是单纯按
连接度截断，而是确定性地兼顾结构重要性、谓词、来源文档、外围分支和直接子社区覆盖；不会
因此增加模型调用。父社区还会收到直接子社区的主题简介作为组织线索，但每个子主题都绑定到
父报告上下文中的原始关系编号，模型不能把子社区简介当成独立证据。输入上下文有实体数、
关系数和 Token 三重边界，但内容哈希覆盖社区全部事实。

报告必须提供可识别主题的语义标题，并为每项结论引用关系编号。质量门禁会检查未知或空引用、
重复结论以及丰富上下文中明显过窄的关系/谓词覆盖，不合格结果在有限次数内重试。社区内容、
模型、Prompt 和配置哈希完全一致时可复用旧报告；图谱、层级上下文或社区事实变化后不会把旧
说明挂到新版本。

目录和社区选择器会在报告完成后同时显示社区编号与语义标题；目录卡片展示简短简介和报告摘要，
进入社区后展示完整发现、局限及引用的原文件/页码；有关联
的图片和表格可从来源卡片打开。报告仍在生成时显示进度占位。联合检索始终先依赖知识块、实体
和两跳关系；已完成报告附加参与词法与向量排序，因此任务失败或尚未完成不会造成知识库不可查。

```dotenv
KF_COMMUNITY_REPORTS_ENABLED=true
KF_COMMUNITY_REPORTS_AUTO_BUILD=true
KF_COMMUNITY_REPORT_INITIAL_LIMIT=12
KF_COMMUNITY_REPORT_SEARCH_ACTIVATION_LIMIT=2
KF_COMMUNITY_REPORT_MIN_ENTITIES=2
KF_COMMUNITY_REPORT_MIN_RELATIONS=1
KF_COMMUNITY_REPORT_MAX_ENTITIES=24
KF_COMMUNITY_REPORT_MAX_RELATIONS=36
KF_COMMUNITY_REPORT_CONTEXT_TOKENS=8192
KF_COMMUNITY_REPORT_MAX_OUTPUT_TOKENS=1800
KF_COMMUNITY_REPORT_MAX_ATTEMPTS=2
KF_COMMUNITY_REPORT_SEARCH_TOP_K=5
```

## Schema 策略

HugeGraph 物理 Schema 固定为：

```text
Vertex: KFDocument, KFChunk, KFEntity
Edge:   kf_contains, kf_mentions, kf_related
```

领域差异保存在 PostgreSQL `graph_schemas.schema_json` 中。每个项目都有独立、
可追溯的逻辑 Schema 版本。Knowledge Forge 的文档预处理只生成结构、知识块和向量；
同一次项目构建中的全部目标文档准备完成后，系统读取并冻结一次活动 Schema。普通构建
默认不执行 LLM Schema 发现：有精确端点、原文关系表达和证据 span 的事实可先作为
`OPEN_GROUNDED` 发布，再尽可能投影到已有规范谓词。这样逻辑 Schema 是归一和治理层，
而不是事实存在与否的前置门槛。完整发现由项目统计区的“重新评估 Schema”独立执行。

重新评估不会固定读取文档开头：系统先过滤目录、封面和过短知识块，再按文档和章节
分层，并利用已生成的 Embedding 贪心选择语义多样的代表块。代表块在全局块数/字符
上限内分批交给 LLM，候选随后统一合并、语义去重，并且整批只应用一次实体/关系新增
数量限制。候选必须引用真实知识块证据，并通过置信度、证据多样性和独立策略核验；
因此多文件结果不再因上传顺序或逐文件重置新增预算而漂移。

已有项目文档可覆盖新文档内容时，系统根据跨文档向量新颖度跳过 Schema LLM 评审，
减少重复调用。自动演进始终只允许追加实体类型和关系类型，不会自动删除、重命名或
改变已有类型含义。代表块数量会随有效知识块数、文档数和章节数做次线性增长，并受块数、
总字符数和单块字符数三重上限保护。重新评估的项目批次请求大小由
`SCHEMA_CORPUS_REVIEW_BATCH_SIZE` 控制；采样基线与上限、新颖度阈值和证据门槛可通过
`SCHEMA_SAMPLE_*`、`SCHEMA_REVIEW_*`、`SCHEMA_MAX_*_ADDITIONS` 配置。

`SCHEMA_DISCOVERY_DURING_BUILD=false` 是默认的低延迟、召回优先模式。只有明确接受
Schema 治理阻塞构建时才应设为 `true`；无论该开关如何，独立重新评估任务和追加式
版本约束都保留。

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
包含关系、缩写、类型、编号/版本冲突和向量分数，再由独立 HTTP 服务中的
`bge-reranker-v2-m3`
交叉编码器重排少量候选。缩写/全称只是名称形态特征之一，不是专用合并规则。
名称、向量和重排候选数量会根据项目中同类型实体规模按对数级增长，并分别受
`ENTITY_LEXICAL_MAX_K`、`ENTITY_VECTOR_MAX_K` 和 `ENTITY_RERANKER_MAX_K` 保护。

重排分数表示语义相关度，不直接等同于实体身份概率。自动合并必须同时满足强名称证据、
类型兼容、无编号冲突和综合阈值；语义相关但身份不确定的候选交给项目 LLM 做最终裁决，
无可靠候选时创建新实体。这样可覆盖人物、组织、产品、地点、标准、事件等不同领域，
同时避免把上下级组织、同类产品或地理包含关系误合并。

每次决策都会写入 `entity_resolution_decisions`，保存候选实体、各项分数、决策方式、
模型和原因。LLM 调用失败或结果不满足接受阈值时默认保留为独立实体，不会冒险合并。
Top-K、召回阈值、重排服务地址和裁决阈值可通过 `ENTITY_*` 环境变量调整。独立服务使用
Transformers 离线推理；Knowledge Forge 主进程不再导入 torch 或持有该模型。Ollama
仍只负责 LLM 与 Embedding。默认模型目录为宿主机
`/home/kdfy/models/bge-reranker-v2-m3`，Compose 只读挂载到 reranker 容器；服务超时、
模型缺失、加载失败或 GPU OOM 时，实体归一保持原有规则、向量和 LLM 回退。

### 独立 Reranker HTTP 服务

用户级 systemd 部署可安装并启动两个进程：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/knowledge-forge-reranker.service ~/.config/systemd/user/
cp deploy/systemd/knowledge-forge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now knowledge-forge-reranker.service
systemctl --user restart knowledge-forge.service
```

本机应用通过 `ENTITY_RERANKER_URL=http://127.0.0.1:8012` 调用。若需要允许可信局域网
机器共享同一个模型实例，将 `RERANKER_HOST` 设为 `0.0.0.0`，并配置至少 16 字符的
`RERANKER_API_KEY`。健康检查不需要凭据：

```bash
SERVER_IP=192.168.14.156
curl "http://$SERVER_IP:8012/health"
```

其他机器通过 Bearer Token 调用；返回结果按相关度排序，`index` 始终指向原始
`documents` 下标：

```bash
SERVER_IP=192.168.14.156
RERANKER_API_KEY='从服务端安全取得的密钥'
curl "http://$SERVER_IP:8012/api/rerank" \
  -H "Authorization: Bearer $RERANKER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "世界卫生组织",
    "documents": ["世界贸易组织", "World Health Organization"],
    "top_n": 2
  }'
```

多台机器可以同时调用：HTTP 层并发接收请求，单个模型实例在服务内部串行执行推理，
每个请求内部继续按 `ENTITY_RERANKER_BATCH_SIZE` 批处理，因此不会重复加载模型，也不会
发生不同请求分数串位。默认最多保留 32 个在途请求、每次最多 128 个候选；超限或排队
超时会返回明确错误。Knowledge Forge 对短暂故障保留有限重试和保守回退，但长期不可用
会降低歧义实体归一质量，应监控 `/health`。非回环监听若没有密钥会拒绝启动，除非显式
设置仅供受信开发网络使用的 `RERANKER_ALLOW_UNAUTHENTICATED_LAN=true`。

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
- `knowledge_forge_community_report_runs`、`knowledge_forge_community_reports`、`knowledge_forge_community_report_attempts`
- `jobs`、`models`、`settings`

Knowledge Forge 不依赖 RAGFlow；项目、文档、知识块和向量均由本项目自己的 PostgreSQL 表管理。

## 当前知识构建架构

### 两种项目级策略

| 策略 | 主要产物 | 存储与查询边界 |
| --- | --- | --- |
| Knowledge Forge | 规范实体、已验证关系、HugeGraph 图、社区与详细报告 | PostgreSQL 是权威事实层，pgvector 提供检索索引，HugeGraph 提供可遍历副本 |
| GraphRAG Native | GraphRAG 3.1.1 实体、关系、社区、报告与查询索引 | 每次运行保存为不可变 Parquet/LanceDB bundle，不写 KF 实体表、pgvector 或 HugeGraph |

源文档、结构块和知识块可被两种策略复用；任何新增、删除或 revision 变化都会使相关图谱进入待更新状态。两种策略必须分别完成新鲜度校验和原子发布，失败构建不会覆盖上一份可查询结果。

### Knowledge Forge 主链

1. 复用已准备的文档结构、知识块和向量；
2. 从活动文档快照确定项目逻辑 Schema，并仅做有证据的增量扩展；
3. 按精确来源 span 抽取实体 occurrence，经过名称、别名、pg_trgm、pgvector、reranker 和必要的 LLM 裁决形成规范实体；
4. 从正文、列表、表格和结构角色生成关系候选，再执行谓词绑定、端点方向检查、独立验证和发布审计；
5. 只有携带当前策略版本、完整证据和可发布终态的关系进入项目图；
6. PostgreSQL 快照验证通过后同步 HugeGraph，数量、端点、方向和代次一致才标记为 `ready`；
7. 社区与详细报告在图谱发布后异步生成，不阻塞 Chunk、实体和关系检索。

关系分类和最终验证都使用有界上下文、候选数、重试和 Token 预算。未覆盖、协议失败、证据不足和冲突均进入明确终态，不会被静默改成通用关系。表格、列表和标题层级只在解析器能够给出精确结构契约时参与确定性关系恢复。

### 图代次、任务与恢复

项目知识变化会递增图代次并把图标记为待更新。连续文档更新会合并到同一目标代次，避免每个文档都触发一次完整图同步。构建、社区报告、图同步和清理均使用 PostgreSQL `jobs` 与持久化工作账本；服务重启后从数据库状态恢复，不依赖进程内存。

实体合并、拆分和撤销保留 occurrence、assignment、redirect 与事件审计，不改写原始来源。文档删除只移除该文档的派生贡献，仍被其他文档支持的实体和关系继续保留。

### 数据库命名兼容

早期迁移创建的 `entity_shadow_*`、`shadow_relation_*`、`relation_shadow_*` 表名、索引、运行 ID 和策略版本值是现有数据的物理兼容契约，不能在普通代码清理中重命名。当前 Python 模块、类、函数和脚本统一使用 `grounded entity`、`grounded relation`、`evidence context` 与 `relation pipeline` 术语。详见 [`docs/naming-and-compatibility.md`](docs/naming-and-compatibility.md)。

### 质量评估与诊断

工程 fixture、跨领域基准和评测工具用于回归抽取完整性、实体边界、归一、谓词、方向、来源和成本：

```bash
.venv/bin/python benchmarks/benchmark_kg_construction.py
.venv/bin/python benchmarks/build_annotation_bundle.py --help
.venv/bin/python benchmarks/evaluate_kg_gold.py --help
.venv/bin/python scripts/export_relation_evaluation_bundle.py --help
.venv/bin/python scripts/validate_relation_evaluation_bundle.py --help
```

`artifacts/` 是可重复生成的本地输出，不提交 Git。模型生成标签只作为待复核候选；生产门禁需要独立确认的 Gold/holdout 与完整分母。默认测试不访问外部模型或图数据库，真实集成测试必须显式使用隔离数据和当前运行模型。

### 智能问答

首页“智能问答”进入统一工作台。当前版本由用户明确选择一个知识库，不自动跨库，也不引入循环工具调用或长期记忆。KF 与 GR 共享会话、证据、引用、停止和反馈协议，但各自只读取所属策略的当前成果。

每轮问题先进入 LangGraph 问题控制子图：高置信度的智能体身份或能力问题直接分流；普通问题从 PostgreSQL 读取同一知识库、同一会话中已经完成的消息，再按当前本地模型的 Token 上下文预算保留最近完整轮次，并把更早内容压缩为滚动摘要。预算会先扣除系统提示、JSON Schema、当前问题、模型输出、安全余量以及一次结构化输出修复的空间；短消息可以保留更多轮，长消息会更早触发压缩，不再固定保留 8 条。结构化 LLM 随后判断问题是否独立、能否补全或必须澄清。补全后的问题只用于召回、重排和答案组织；历史回答仅用于识别“它、前者、上述方法”等指代，永远不作为知识证据。原始问题、独立问题、解析引用、Token 窗口指标和分支结果均保存在运行审计元数据中。

问答运行采用访问、输入、对话、检索、执行和输出六层策略协议。LLM 只承担多轮语义解析和边界证据对齐；项目权限、KF/GR 工具白名单、调用上限、快照新鲜度、来源有效性和声明引用完整性由代码强制执行。越界或没有直接证据的问题不会用通用知识补答，歧义问题不会带着猜测进入检索。每层结果写入运行审计，并在“执行过程”中显示安全摘要，不暴露系统提示或隐藏思维链。

LangGraph 使用官方 PostgreSQL checkpointer，并把执行状态隔离在 `knowledge_agent` schema。`agent_messages` 仍是会话事实的权威来源，checkpoint 只保存有界窗口、滚动摘要和流程位置；归档会话时会清理对应 checkpoint。第一轮没有历史时不会额外调用语境模型，后续轮次最多调用一次语境解析模型。

问答运行、阶段、工具调用、证据、逐项引用、SSE 事件和反馈均持久化到 PostgreSQL。每条回答都可展开“执行过程”：运行中实时显示问题理解、检索选择、证据检查、补充检索、回答与引用核验等阶段，以及实际调用的检索、reranker 和回答工具；每个步骤或工具还可继续展开，查看其白名单化的输入范围、结果计数、模型、耗时和约束。召回、重排、充分性检查和回答阶段会列出最终进入上下文的具体证据；引用核验阶段只列出最终实际引用的证据，并区分“生成依据”与“最终引用”。证据卡提供类型、标题、文件、页码/章节和有界摘录，点击后复用授权来源详情查看完整原文、图片和表格。历史会话刷新后仍可从运行台账恢复。该视图是现有结构化台账的确定性安全投影，不增加 LLM 调用，也不暴露隐藏思维链、系统提示词、原始工具载荷或未验证草稿。Local、Global、Basic 只在 GraphRAG Native 知识库中显示；Knowledge Forge 始终使用自身的动态混合检索规划。浏览器使用不可读的能力 Cookie 隔离会话；来源只暴露受控下载地址。服务重启会把未完成回答标记为失败，不会盲目续写旧快照答案。

主要接口：

```text
GET    /api/agent/knowledge-bases
GET    /api/agent/threads?project_id={project_id}
POST   /api/agent/threads
GET    /api/agent/threads/{thread_id}
DELETE /api/agent/threads/{thread_id}
POST   /api/agent/threads/{thread_id}/messages
GET    /api/agent/runs/{run_id}
GET    /api/agent/runs/{run_id}/trace
GET    /api/agent/runs/{run_id}/events
POST   /api/agent/runs/{run_id}/cancel
POST   /api/agent/runs/{run_id}/feedback
GET    /api/agent/evidence/{evidence_id}
```

运行上限通过 `KNOWLEDGE_AGENT_*` 环境变量配置；每次运行会固化自己的候选数、语境解析、证据上下文、输出和调用次数预算。多轮语境窗口先取知识库配置与 Ollama 实际 runner 上下文的较小值，再按本轮完整提示的估算 Token 从 `KNOWLEDGE_AGENT_CONTEXT_RESOLUTION_MIN_BUDGET_TOKENS` 起，以 `*_GROWTH_STEP_TOKENS` 为阶梯扩容；动态上限同时受 `*_MAX_BUDGET_TOKENS` 和 `*_MODEL_RATIO` 约束。默认在 qwen3.6 的 262144 Token runner 上按需使用 8192～32768 Token，完整历史能放入动态窗口时不会提前压缩，超过上限后才由 `KNOWLEDGE_AGENT_CONTEXT_RECENT_TOKEN_RATIO` 划分最近原始轮次与增量压缩空间，`KNOWLEDGE_AGENT_CONTEXT_SUMMARY_MAX_TOKENS` 限制滚动摘要。旧的 `KNOWLEDGE_AGENT_CONTEXT_RESOLUTION_BUDGET_TOKENS` 仅作为兼容配置：当新最大值未设置时继续充当上限。`*_MIN_RECENT_TURNS`、`*_RECENT_MAX_MESSAGES`、`*_BOOTSTRAP_MESSAGES` 和字符上限只承担极端输入及数据库读取保护，不再决定日常压缩时机。模型上下文上限低于 `KNOWLEDGE_AGENT_MIN_CONTEXT_TOKENS` 的知识库不会进入问答队列。无答案判断不会采用相对排序分单独决策，而是联合交叉编码器绝对相关度、Embedding 相似度和问题要点覆盖率；三个门槛可分别通过 `KNOWLEDGE_AGENT_WEAK_RERANK_THRESHOLD`、`KNOWLEDGE_AGENT_WEAK_SEMANTIC_THRESHOLD` 和 `KNOWLEDGE_AGENT_WEAK_ASPECT_THRESHOLD` 校准。

问答范围由前后两道独立门禁控制。身份、能力、使用方式和简单问候等智能体交互问题先由高精度通用规则识别；其余问题在完成多轮指代补全后，由当前项目 LLM 结合知识库名称、说明和文档范围做一次只分类、不作答的检索前语义路由。高置信度的知识库外问题直接返回范围说明，不再执行召回和 reranker；低置信度、范围资料不足或分类异常时保留检索，以免降低知识问题召回率。进入检索的问题在召回与重排后还要确认问题和证据属于同一知识范围：强交叉编码器相关度，或同时满足语义相关与问题要点覆盖的结果可直接通过；边界结果再次由结构化模型核验。无关或无直接支持的问题不进入事实回答模型，也不使用通用知识补答。检索前跳过门槛由 `KNOWLEDGE_AGENT_INPUT_SCOPE_CONFIDENCE_THRESHOLD` 配置，证据范围门槛与 Token 上限通过 `KNOWLEDGE_AGENT_SCOPE_*` 配置。

Knowledge Forge 社区导航由 `KNOWLEDGE_AGENT_COMMUNITY_NAVIGATION_ENABLED` 控制，默认最多扩展 3 个社区联系、回取 8 条底层关系。它只在多社区、关系稀疏或可靠社区报告种子场景启动，不增加 LLM/Embedding 调用；任何候选都必须再次通过当前关系快照和来源门禁。

动态检索计划不额外调用模型：它用通用中英文问题形态选择意图和证据族预算，但始终给各证据族保留安全下限。带明确编号的图表问题会从当前文档 revision 中恢复对应 Chunk 和媒体来源；证据不足时最多执行一次缺口驱动的补充计划，随后必须回答、限定回答或拒答。GR 自动模式对单文档总结使用 Local，对真正的多文档/项目级总结使用 Global；用户显式指定 Basic/Local/Global 时保持原选择。Global 查询依据项目模型的实际上下文窗口调整社区报告批次，不修改不可变 Bundle。

验证工程基准文件并调用真实 HTTP/Ollama 链路：

```bash
.venv/bin/python benchmarks/knowledge_agent_v1/validate.py
.venv/bin/python benchmarks/knowledge_agent_v1/evaluate.py \
  --project-id PROJECT_ID --assert-acceptance
```

当前冻结的 28 题中英双语工程校准集（含 4 题无答案）在 2026-08-24 的真实链路验收中通过全部门禁：任务、引用解析、声明引用完整和拒答均为 100%，答案要点召回为 92.43%，跨项目证据为 0。该数值只用于当前固定语料的回归校准，不作为开放领域效果承诺。

当前智能体持久化由两个不可变迁移共同提供：`0035_knowledge_agent_v1` 建立会话、运行、消息、步骤、工具、证据、引用、事件与反馈账本；`0036_knowledge_agent_context_runtime` 为 LangGraph PostgreSQL checkpointer 建立隔离 schema。当前版本仍是显式单知识库问答，下一阶段从显式多选知识库及跨库证据隔离开始。

完整架构、证据协议、LangGraph 边界、评估门槛和后续阶段见 [`docs/knowledge-agent-implementation-roadmap.md`](docs/knowledge-agent-implementation-roadmap.md)。

## 验证

上线前 revision、不可变构建快照、Gold/release gate、HugeGraph sentinel、迁移恢复和隔离
E2E 的完整操作见 [`docs/pre-live-readiness.md`](docs/pre-live-readiness.md)。

```bash
.venv/bin/python -m py_compile knowledge_forge/*.py server.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_disposable_postgres_tests.py
curl http://127.0.0.1:8011/health
# 按当前 .env 中的 HUGEGRAPH_HOST/HUGEGRAPH_PORT 替换；本机当前为 127.0.0.1:8081
curl http://127.0.0.1:8081/graphs
```

当前自动化测试的通过数量以本机执行结果为准。Docker 首次构建需要本机已有
`python:3.12-slim`，或能够访问镜像仓库下载该基础镜像。
