# Knowledge Forge 当前知识库与知识图谱策略

> 端到端技术说明、算法细节与面试手册
>
> 基于仓库截至 2026-08-15 的当前实现整理。本文描述的是 <code>knowledge_forge</code> 策略，不是独立的 <code>graphrag_native</code> 策略。

## 0. 文档定位与阅读方式

这份文档有两个目标：

1. 准确说明 Knowledge Forge 从用户上传文档，到知识库检索和知识图谱可用的完整执行链。
2. 帮助项目负责人在面试中解释设计动机、算法、工程取舍、质量指标、失败闭环和后续演进，而不只是罗列功能。

需要先说明三个边界：

- 当前产品策略名是 <code>knowledge_forge</code>。数据库中部分 <code>shadow_*</code> 表名、ID 前缀和内部模块名是历史兼容名称，不代表线上仍运行一套名为 “Shadow” 的产品策略。
- <code>graphrag_native</code> 是另一条完全独立的 GraphRAG 3.1.1 路线。它使用自己的 Parquet/LanceDB 运行包，不经过本文的实体归一、关系审核、PostgreSQL 事实快照或 HugeGraph 物化链。
- 当前策略元数据仍是 <code>production_approved=false</code>。本文中的评测结果是工程验收证据，不应在面试中表述成经过第三方认证的生产级准确率。

本文生成时已使用项目 <code>.venv</code> 完成代码自检：Python 编译通过，测试结果为 1163 passed、45 skipped；跳过项是测试配置中显式标记的可选集成场景。

推荐阅读顺序：

- 面试前快速复习：第 1、2、23、25、26、27 节。
- 深挖算法：第 7 至 15 节，以及第 18 节。
- 深挖工程可靠性：第 4 至 6、16、20 至 22 节。
- 深挖 RAG 与图谱协同：第 9、18、19 节和第 23.10 节。

章节导航：

- 第 1–3 节：项目定义、设计目标和总体架构。
- 第 4–11 节：用户操作、上传、解析、分块、向量和 Schema。
- 第 12–15 节：来源范围、实体抽取、实体归一和关系发布。
- 第 16–19 节：图物化、社区、混合 RAG 和前端图浏览。
- 第 20–24 节：数据模型、可靠性、复杂度、质量与已知局限。
- 第 25–31 节：设计取舍、项目叙述、面试问答和速查表。

---

## 1. 一句话定义与面试摘要

### 1.1 一句话定义

Knowledge Forge 是一个面向多项目、多领域文档的知识库与知识图谱构建系统：它将原始文件解析为带精确来源坐标的结构块，在同一份内容修订上同时构建向量知识库和经过独立证据审核的实体关系图，并用版本化、可取消、可回滚的快照机制发布给混合 RAG。

### 1.2 30 秒面试版

传统 “让大模型从分块里直接输出三元组” 的主要问题是关系召回和可信度难以兼得：分块会丢结构，模型会把共现误判成关系，实体别名会重复，长文档还会产生大量重复调用。

我的方案把流程拆成三层：

1. **候选层**：结构解析、实体清单、开放事实清单和确定性结构规则尽量提高召回。
2. **证据层**：每个实体和关系必须绑定原文精确区间、文档、修订、块和原子语义范围。
3. **事实层**：独立验证器检查端点角色、方向、类型、否定和蕴含，只有通过发布门槛的事实才进入图谱。

同时，所有文本块都进入向量知识库，而目录、父级摘要、参考文献等只参与检索、不参与图谱抽取。最终 RAG 同时使用原文块、实体和一到两跳的已验证关系，图谱是检索增强源之一，不替代原始知识库。

### 1.3 3 分钟面试版

完整链路可以概括为：

1. 上传时校验文件类型和大小，计算 SHA-256，做项目内内容去重，并保存不可变源文件。
2. 预处理阶段自动选择 DOCX、文本、PDF、Docling 或表格解析器，产出保留章节、列表、表格、页码和图像来源的结构块。
3. 基于结构做分块。所有块都可进入向量检索，但只有正文、列表项、表格行和媒体描述等可验证内容进入图谱抽取。
4. 对每次内容、配置、模型和策略计算指纹，预处理结果先写入暂存修订，成功后原子激活；失败不会破坏上一版。
5. Schema 分成固定物理图模型和版本化逻辑 Schema。普通构建默认冻结当前 Schema，避免 Schema 发现阻塞召回；无法映射到固定谓词但原文明确表达的事实，可通过开放关系载体保留其原始谓词。
6. 实体阶段先让模型做来源约束的 mention inventory，再用精确重复恢复和结构候选补召回。所有 mention 都必须能回到原文区间。
7. 实体归一先做精确别名和硬冲突判断，再组合词法、向量和 reranker 分数；非精确候选即使分数高，也要经过有界身份复核，不确定就新建实体，避免错误合并。
8. 关系阶段先按原子来源范围枚举开放事实，再用列表、表格、标签、并列等通用结构投影补候选。之后做 Schema 绑定、精确证据校验和独立有序蕴含验证。
9. 归一化层消除重复、逆关系和对称关系冲突。只有端点、证据、方向、类型、谓词发布模式、快照新鲜度全部通过的事实才能物化。
10. PostgreSQL 保存控制面、不可变审计和检索快照；HugeGraph 保存项目隔离的可浏览物理图。图发布使用 generation、lease 和 CAS，构建失败或安全终止时仍保留上一可用版本。
11. 社区检测只借用 GraphRAG 的 Leiden 聚类，不借用其抽取流程。社区摘要由已验证实体、关系和原文证据确定性生成。
12. 查询时将文本块检索、实体检索和最多两跳的图关系召回融合，并做事实级去重和多样性控制。

---

## 2. 解决的问题与核心设计原则

### 2.1 系统要解决的实际问题

文档知识图谱不是简单的命名实体识别加三元组生成。通用场景至少有以下困难：

- 文档格式异构：PDF、DOCX、Markdown、纯文本、Excel/CSV 的结构信号不同。
- 中文技术文档常用标题、缩进列表、表格、括号枚举表达关系，句法不一定完整。
- 长文档分块后，标题与子项、表头与单元格、跨页列表之间的结构容易丢失。
- 同一个实体可能有全称、简称、大小写、符号和版本差异。
- 两个实体同段出现不等于存在关系。
- 关系方向经常受主动句、被动句、倒装、列表结构影响。
- Schema 太窄会损失召回，Schema 太宽又容易产生无意义的 “相关”。
- 大模型调用可能超时、截断、返回非法 JSON，不能把技术失败当成事实否定。
- 长任务需要在服务重启、用户终止和代码升级后保持可解释状态。
- 图谱最后是 RAG 的一个来源，因此不能只追求图的边数，还要兼顾检索价值、证据可读性和成本。

### 2.2 七条核心原则

#### 原则一：来源先于语义

模型可以建议 “是什么实体、是什么关系”，但不能创造来源。每条 mention、候选关系和已发布事实都要保存文档、内容修订、块、来源范围和字符区间。

#### 原则二：检索内容与图谱事实分流

适合回答问题的文本不一定适合生成事实。目录、父级摘要和参考文献可用于 RAG，但不会因为包含大量术语就自动生成图关系。

#### 原则三：候选、证据、事实分层

- 候选层允许高召回和不确定性。
- 证据层要求精确可定位。
- 事实层要求独立验证和发布策略通过。

因此，模型置信度不是最终真值分数。

#### 原则四：Schema 是治理层，不是召回闸门

物理图 Schema 固定，逻辑 Schema 可版本化、只增不减。当前 Schema 没有合适谓词时，原文明确的二元事实仍可用开放载体保存，避免为了适配 Schema 把真实语义硬塞进 <code>related_to</code>。

#### 原则五：高风险步骤默认 fail closed

实体身份不确定就保持分离；关系方向不确定就不发布；模型响应截断或协议错误就重试或失败，不将其解释为 “没有关系”。

#### 原则六：所有可见结果都来自新鲜、完整快照

内容、模型、代码、Schema 和策略变化都会进入指纹。图谱只有在完整构建、验证和 CAS 激活后才可查询。

#### 原则七：图谱增强 RAG，而不是取代原文

图关系适合提供实体连接、因果链和结构线索；原始文本块适合保留限定条件、上下文和长答案。最终检索必须融合两者。

---

## 3. 总体架构

### 3.1 端到端数据流

~~~mermaid
flowchart LR
    U[用户上传文件] --> H[扩展名/大小/SHA-256 校验]
    H --> S[不可变源文件与文档记录]
    S --> P[解析器路由]
    P --> B[结构化 DocumentBlock]
    B --> M[图片/OCR/视觉描述]
    B --> C[结构感知分块]
    M --> C
    C --> V[向量与词法索引]
    C --> G0[仅选择图谱可抽取块]
    G0 --> SC[冻结或显式演进逻辑 Schema]
    SC --> CTX[原子来源范围与精确 source map]
    CTX --> EI[实体 mention inventory]
    EI --> ER[实体归一与 canonical entity]
    ER --> RI[开放事实清单与结构候选]
    RI --> SB[Schema 绑定]
    SB --> RV[独立方向/角色/证据验证]
    RV --> NF[归一事实与冲突消解]
    NF --> PG[(PostgreSQL 不可变事实快照)]
    NF --> HG[(HugeGraph 可浏览图)]
    NF --> COM[Leiden 社区与确定性描述]
    V --> RAG[混合 RAG]
    PG --> RAG
    COM --> RAG
~~~

### 3.2 运行时职责

| 组件 | 职责 | 不负责的内容 |
|---|---|---|
| PostgreSQL | 项目、文档、修订、块、向量、Schema、实体、关系、审计、任务、图版本和社区控制面 | 大规模交互式图遍历渲染 |
| HugeGraph | 固定 <code>KF*</code> 物理 Schema 下的项目隔离图、实体关系浏览 | 作为关系真值和模型审计的唯一来源 |
| Ollama | 模型发现、嵌入、Schema 审查、实体清单、身份复核、关系清单与独立验证 | 持久任务状态或最终发布决策 |
| <code>KnowledgeFiles/</code> | 原始文档、派生 chunk JSON、媒体和软删除回收站 | 业务状态的唯一记录 |
| Web/API | 用户操作、状态展示、图谱探索、检索入口 | 实体关系领域算法 |

### 3.3 当前策略与 GraphRAG Native 的边界

| 维度 | Knowledge Forge | GraphRAG Native |
|---|---|---|
| 抽取方式 | 来源约束实体 + 开放事实 + 独立验证 | GraphRAG 3.1.1 原生流程 |
| 实体归一 | PostgreSQL 中的保守 canonical resolution | 原生运行包内部产物 |
| 关系真值 | 不可变关系运行、事实和证据表 | Parquet/LanceDB bundle |
| 物理图 | HugeGraph <code>knowledge_forge</code> | 不进入 HugeGraph |
| 社区 | 仅借用 Leiden 检测 | 原生社区与报告 |
| 回退 | 失败时禁止静默回退 | 失败时也禁止切到 KF |

二者是项目级策略选择，不是同一条流水线中的两个阶段。

---

## 4. 用户操作、任务状态与安全终止

### 4.1 项目创建

用户创建项目时确定：

- 项目名称和隔离键 <code>project_key</code>。
- 构建策略，当前主要为 <code>knowledge_forge</code> 或 <code>graphrag_native</code>。
- LLM 模型，默认 <code>qwen3.6:latest</code>。
- Embedding 模型，默认 <code>bge-m3:latest</code>。
- 项目级模型上下文预算、chunk 上限和策略配置。

策略属于项目不可变边界。系统不会在一次失败后偷偷换用旧流程。

### 4.2 上传、预处理和构建是三个不同动作

| 动作 | 主要产物 | 是否调用图谱抽取 |
|---|---|---|
| 上传 | 源文件、文档元数据和内容哈希 | 否 |
| 文档预处理 | sections、chunks、媒体描述、embeddings、活动内容修订 | 否 |
| 知识图谱构建 | Schema 快照、实体命名空间、关系运行、图 generation、社区 | 是 |

这样拆分的原因是：

- 上传可以快速返回。
- 内容解析与图谱算法可独立重试。
- 修改关系代码时可以复用未变化的内容修订和实体结果。
- 向量知识库可以先准备好，不必等待完整图构建。

### 4.3 持久任务

长任务写入 PostgreSQL <code>jobs</code> 表，典型状态为：

- <code>queued</code>
- <code>running</code>
- <code>completed</code>
- <code>failed</code>
- <code>cancelled</code>

任务进度、日志、错误、元数据和取消请求都持久化，不依赖 Python 进程内存。

当前普通长任务执行器采用单 worker。它牺牲一部分并发吞吐，换取：

- 本地模型和 GPU 不被多个大型任务争抢。
- 项目级构建顺序确定。
- 日志和资源峰值更容易解释。
- 安全终止后不会立即与替代任务竞态。

### 4.4 安全终止不是强杀进程

用户点击终止时：

1. 数据库记录持久化 <code>cancel_requested=true</code> 和请求时间。
2. 正在运行的 worker 在解析批次、模型调用、写入批次和图同步等检查点读取取消状态。
3. worker 停止继续生产新结果。
4. 尚未激活的内容修订或图 generation 被放弃。
5. 任务进入 <code>cancelled</code>。
6. 上一版活动内容修订继续可用；上一图 generation 的物理产物不会被未完成结果覆盖，但当前公开图路由会将 stale/failed 图封闭，直到一次完整重建重新发布。

不能把终止实现成简单 kill，因为那会留下：

- 半写入的 chunks；
- 已更新 PostgreSQL 但未更新 HugeGraph 的混合状态；
- 永久处于 building 的 lease；
- UI 看见并不存在的完成进度。

### 4.5 图状态

项目策略图主要经历：

~~~text
empty -> stale -> rebuilding -> ready
                    |           |
                    +-> failed  +-> stale（源内容或策略发生变化）
~~~

- <code>empty</code>：从未有可用图。
- <code>stale</code>：历史 ready generation 仍被保留，但公开查询拒绝把它当作当前图。
- <code>rebuilding</code>：正在构建新 generation。
- <code>ready</code>：完整快照已通过发布 CAS。
- <code>failed</code>：构建失败且没有成功激活本次 generation；历史 generation 不会被半成品覆盖，但当前图查询被封闭。

取消重建时，如果已有 ready generation，状态回到 stale；如果从未成功构建，则回到 empty。这里的 “保留上一版” 指可恢复、可审计的历史产物，不代表 stale 状态仍会由当前公开查询接口返回。

---

## 5. 上传、内容寻址与文档生命周期

### 5.1 上传校验

当前允许的后缀：

- <code>.docx</code>
- <code>.pdf</code>
- <code>.md</code> / <code>.markdown</code>
- <code>.txt</code>
- <code>.xlsx</code> / <code>.xls</code>
- <code>.csv</code>

默认上传上限为 200 MB，可通过环境变量调整。

上传流程：

1. 校验扩展名、请求解析器、chunk 策略和项目策略。
2. 写入临时文件。
3. 流式计算 SHA-256：

~~~text
document_hash = SHA256(file_bytes)
~~~

4. 拒绝同一项目内仍有效的完全重复内容。
5. 用内容哈希前缀和清洗后的文件名生成存储名：

~~~text
stored_name = hash[0:16] + "_" + safe_stem + suffix
~~~

6. 写入文档记录，包括：

   - <code>project_id</code>
   - <code>project_key</code>
   - 文件哈希、大小、MIME 和存储路径
   - 请求解析器
   - chunk 策略和配置
   - 文档状态

### 5.2 为什么做内容寻址

内容哈希同时解决：

- 上传去重；
- 文件名变化但内容相同的识别；
- 内容修订的新鲜度判断；
- 预处理缓存复用；
- 构建输入指纹；
- 审计时证明 “事实来自哪一版字节”。

### 5.3 软删除与重新摄取

文档删除采用软删除：

- PostgreSQL 中保留文档和来源链。
- 源文件移入 <code>KnowledgeFiles/.trash/</code>。
- 只撤销该文档对派生数据的贡献。
- 如果同一事实仍有其他文档证据，不删除该事实。

相同哈希的软删除文档可以恢复，而不是创建一套无法追踪的新身份。

重新摄取遵循 “只替换选中文档的派生贡献” 原则，不能误删同项目其他文档支持的实体和关系。

---

## 6. 内容修订、指纹与原子激活

### 6.1 为什么需要内容修订

如果解析中途直接覆盖线上 sections/chunks，一旦 PDF 解析、OCR、embedding 或数据库写入失败，用户会得到半新半旧的数据。

因此，每次预处理先创建不可变 <code>document_content_revisions</code> 暂存修订：

1. 在新 revision 下写 sections、chunks、媒体和 embeddings。
2. 完成一致性检查。
3. 原子切换文档的 <code>active_revision_id</code>。
4. 再清理不再活动的暂存产物。

失败或取消时，上一活动修订保持不变。

### 6.2 修订清单

一个内容修订至少由以下哈希约束：

~~~text
content_hash = SHA256(source_file_bytes)
source_hash  = content_hash
config_hash  = SHA256(canonical_json(preparation_config))
policy_hash  = graph_policy_hash(policy_axes, active_schema_version)
~~~

其中 <code>preparation_config</code> 包括：

- 结构解析 profile；
- 分块 profile；
- Schema 采样 profile；
- 请求解析器和 chunk 策略；
- 项目 chunk 上限；
- 通用、论文和兼容分块参数；
- LLM 与 embedding 模型；
- 影响输出的处理版本。

当前关键 profile 包括：

- <code>kf-document-structure-v8-docling-heading-sanity-all-routes</code>
- <code>kf-document-chunks-v11-pdf-soft-wrap-sentence-integrity</code>
- <code>kf-schema-graph-eligible-v1</code>

### 6.3 复用条件

共享内容预处理可复用要求：

~~~text
active_revision exists
AND content_hash equal
AND source_hash equal
AND config_hash equal
~~~

<code>policy_hash</code> 主要在图层判断新鲜度。这样做允许：

- 关系策略升级时复用完全相同的解析和 chunks；
- Schema 变化时重建图而不是重复解析文件；
- embedding 模型变化时避免混用旧向量空间。

### 6.4 项目级准备调度

项目预处理入口会：

1. 校验项目策略未改变。
2. 计算全部文档当前修订清单。
3. 找出 stale 文档。
4. 为每个 stale 文档创建持久 <code>content_preparation</code> 任务。
5. 由单 worker 串行执行。

如果文档都已新鲜，系统不会因为用户再次点击而重复解析。

---

## 7. 文档解析与结构恢复

### 7.1 统一中间表示

不同解析器最终都转换成 <code>DocumentBlock</code>：

| 字段 | 含义 |
|---|---|
| <code>key</code> | 稳定来源块键 |
| <code>block_type</code> | 标题、正文、列表项、表格行、媒体描述等 |
| <code>text</code> | 原始可验证文本 |
| <code>order</code> | 文档内顺序 |
| <code>section_key</code> | 所属章节 |
| <code>section_title</code> | 章节标题 |
| <code>level</code> | 标题或结构层级 |
| <code>heading_path</code> | 完整父级标题路径 |
| <code>page</code> | 页码或页范围 |
| <code>metadata</code> | 表格、列表、媒体和解析器特有坐标 |

后续分块、来源映射和图抽取只依赖这个统一表示，不直接依赖 PDF 或 DOCX 库的内部对象。

### 7.2 自动解析器路由

| 文件情况 | 首选解析方式 |
|---|---|
| DOCX | 原生 DOCX 解析 |
| Markdown / TXT | 原生文本结构解析 |
| XLSX / XLS / CSV | 表格解析 |
| PDF 无可提取文本 | Docling；不可用则明确失败 |
| PDF 有复杂布局且 Docling 可用 | Docling |
| 普通文本 PDF | 轻量文本 PDF 解析 |

PDF 还有两层质量回退：

1. 自动选择 Docling 后，如果 Docling 失败但 PDF 确实有文本，则退回轻量文本解析。
2. 轻量解析后，如果有效字符数小于：

~~~text
quality_floor = max(200, page_count × 40)
~~~

且 Docling 可用，则改用 Docling 重试。

这是解析器级回退，不是知识抽取策略回退；最终仍进入同一 Knowledge Forge 流程。

### 7.3 文档类型识别

论文特征不是根据文件名硬编码，而是检查摘要、关键词、方法、实验、结论、参考文献等标记。不同标记达到至少 4 类时倾向论文策略。

电子表格进入 table 策略，其余默认 general。

论文与普通文档共用保守的双语镜像识别：只有正文主语言明确、摘要/执行摘要/内容提要/概述存在中英文成对标题、两份内容篇幅相近且位于文档前部时，次语言版本才被标记为仅检索。未成对的英文 Summary、普通英文技术章节和真正的双语正文继续参与图谱抽取。

实体边界审计也不依赖论文结构或文件格式。模型若把带情态词、完整主谓结构、续接连词或时态标记的整句误标成 Event、Organization 等实体类型，该 mention 会保留在审计记录但不进入关系候选和公开图；明确的事件名、名词短语、列表功能项、表格单元格与显式命题参数继续保留。这一确定性投影在模型抽取后、关系候选规划前执行，不增加模型调用。

### 7.4 结构恢复

#### DOCX

保留：

- 标题层级；
- 段落顺序；
- 列表样式和缩进；
- 表格、行和单元格；
- 图片锚点；
- 章节路径。

#### Markdown / 纯文本

识别：

- <code>#</code> 标题；
- 编号标题；
- 项目符号；
- 缩进层级；
- 空行和段落；
- 简单表格。

#### PDF

保留页级来源，并尽量识别：

- 标题；
- 正文；
- 列表项；
- 页眉页脚；
- 页码；
- 表格；
- 图片。

编号标题检测会排除明显的表格行、列间距、全大写代码标识和过长文本，减少把技术参数误当章节标题的情况。

### 7.5 图像、OCR 与视觉描述

系统可以从 PDF 和 DOCX 中提取图片：

1. 归一化为 PNG。
2. 按图像内容哈希去重。
3. 过滤过小图片：

~~~text
width  >= 96
height >= 96
pixels >= 16000
~~~

4. 最大边缩放到不超过 2048。
5. 可选 RapidOCR。
6. 调用多模态 Ollama，要求严格 JSON：

~~~json
{
  "title": "...",
  "description": "...",
  "ocr_text": "...",
  "facts": [],
  "relationships": []
}
~~~

如果 PDF 页没有嵌入图片且文本字符数不超过 40，可将该页渲染成图片，处理扫描页。

图片分析失败只记录该图片错误，不中止整篇文档。生成的媒体描述会成为带来源的 <code>media_description</code> 块，但其中的关系仍只是后续文本证据候选，不能绕过关系验证直接发布。

---

## 8. 结构感知分块

### 8.1 近似 token 公式

为避免依赖每种模型 tokenizer，分块阶段使用保守估计：

~~~text
T_hat(text) =
    max(
        1,
        N_CJK
        + ceil(1.3 × N_ASCII_word)
        + ceil(N_other_punct / 4)
    )
~~~

其中：

- <code>N_CJK</code>：中日韩统一表意字符数；
- <code>N_ASCII_word</code>：英文、数字和下划线词数；
- <code>N_other_punct</code>：其余字符或标点规模。

该估计偏保守，目标是防止 chunk 和模型请求超过实际上下文。

### 8.2 默认分块参数

| 策略 | 目标 token | 最大 token | 重叠 |
|---|---:|---:|---:|
| general | 512 | 800 | 12% |
| paper | 640 | 900 | 10% |
| compatibility | 按字符 2400 | 按字符 2400 | 160 字符 |
| table | 按行并重复表头 | 受项目硬上限约束 | 不做普通滑窗 |

通用文档在章节数达到 24 时，会尝试合并连续微型子节，同时保留内部标题和来源键。环境变量设为 0 可关闭。

### 8.3 超长片段切分

先估计字符/token 比：

~~~text
chars_per_token = max(1, len(segment) / T_hat(segment))
width = floor(max_tokens × chars_per_token × 0.95)
~~~

再尽量按句子边界切分。<code>0.95</code> 是安全余量。

块间只复制完整前序片段作为重叠，重叠预算为：

~~~text
overlap_budget =
    floor(target_tokens × overlap_percent / 100)
~~~

不会在字符中间机械截取一段重叠，从而避免产生不可定位的半句。

### 8.4 通用文档策略

通用文档按章节树处理：

1. 标题建立父子层级。
2. 小章节优先保持完整。
3. 超长章节按句子切成多个 child chunk。
4. 当一个章节产生多个 child 时，额外生成 retrieval-only 的 parent overview。
5. 大量连续微型章节可在预算内合并，但每个内部标题和来源块仍保留。

父级摘要的默认目标约 220 token、最大约 320 token，并设置：

~~~text
chunk_role = "parent"
graph_extraction_enabled = false
~~~

它帮助检索先命中主题，再返回细节；但摘要不是原始事实载体，因此不参与图谱抽取。

### 8.5 论文策略

论文分为：

- 摘要；
- 正文；
- 参考文献。

正文与主语言摘要可进入图谱，参考文献保持可检索但默认不参与图抽取，避免把引用作者、论文标题和出版信息误判成正文事实。参考文献识别兼容全角编号和跨页续排，作者简介等文后资料同样保留检索而不生成正文事实。若论文同时包含结构成对、篇幅相近的中英文摘要，系统依据正文判断主语言；次语言镜像摘要仍进入向量知识库并保留来源信息，但不再重复抽取同一批实体关系。PDF 解析把译文标题、作者、带 OCR 空格的 `Abstract` 或摘要正文拆到相邻结构节乃至下一页时，系统依据同页/邻页、脚本主导语言和关键词边界恢复完整镜像前置内容。该规则不按语言粗略过滤，无法确认镜像关系时保持图谱抽取资格。

双栏期刊 PDF 的末页还可能把“下一篇文章首页”排在“当前文章参考文献续页”之前。系统仅在此前已出现当前文章摘要和参考文献，并且同一后续页同时出现新的摘要、关键词、DOI 以及旧参考文献续接时，把新文章预览标记为仅检索；完整的多文章合集不会仅因出现第二个标题而被隐藏。

### 8.6 表格策略

表格按行生成 chunk，并重复表头，让每一行离开原表后仍有语义。表头、行号、单元格和表名的来源坐标进入 metadata。

表格关系只能来自：

- 同一行或同一单元格的明确角色；
- 表头对列值的语义约束；
- 经过验证的结构投影。

仅仅处于相邻行或相邻列不会直接成为关系。

### 8.7 目录识别

当一个文本块中：

- 至少有 3 行点线引导项；
- 点线引导项占非空行比例至少 0.5；

系统将其识别为目录。目录仍可检索，但不参与实体关系抽取。

### 8.8 图谱抽取资格

| 内容角色 | 向量检索 | 图谱抽取 |
|---|---:|---:|
| 正文 body | 是 | 是 |
| 列表项 list_item | 是 | 是 |
| 表格行 / 单元格 | 是 | 是 |
| 媒体描述 | 是 | 是 |
| 主语言摘要 / 概述 | 是 | 是 |
| 成对的次语言镜像摘要 / 概述 | 是 | 否 |
| 与旧参考文献跨栏交错的下一篇文章预览 | 是 | 否 |
| 父级 overview | 是 | 否 |
| 目录 | 是 | 否 |
| 参考文献 | 是 | 否 |
| 标题本身 | 是 | 通常否；只作为结构上下文 |
| 页眉、页脚、页码 | 是或保留 | 否 |
| 表格容器、表头 | 是或作为上下文 | 不单独抽取 |
| caption、footnote、index | 可保留 | 默认否 |

这是回答 “什么知识块不参与图谱抽取” 的核心：系统不是按字符长度粗暴排除，而是按来源角色和事实可验证性分流。

### 8.9 硬上限

最终每个 chunk 都要通过项目级最大 token 检查。chunk 会带标题、章节 ID、源文档和分片编号等头信息；如果仅头信息就超过上限，系统明确失败，而不是静默截断来源。

---

## 9. 向量索引与文本检索

### 9.1 向量身份

向量记录至少绑定：

- <code>project_id</code>
- 文档和内容修订；
- chunk 或实体；
- embedding 模型名；
- 向量维度；
- 输入内容哈希。

模型名属于向量身份，因此更换 embedding 模型后不会混合两个向量空间。

### 9.2 pgvector 索引

默认使用 HNSW：

| 参数 | 默认值 |
|---|---:|
| <code>m</code> | 16 |
| <code>ef_construction</code> | 64 |
| <code>ef_search</code> | 100 |
| 最大维度 | 2000 |

余弦相似度：

~~~text
cos(q, x) = (q · x) / (||q|| × ||x||)
D = max(0, cos(q, x))
~~~

数据库如果返回余弦距离 <code>1 - cos</code>，应用层再转换为相似度。

### 9.3 词法召回

词法 term 包括：

- 去空格、casefold 后的完整串；
- 长度至少 2 的英文、数字和下划线词；
- 中文重叠 bigram。

使用多重集合 query recall：

~~~text
L =
    sum_t min(count_query(t), count_doc(t))
    / sum_t count_query(t)
~~~

它比简单集合 Jaccard 更重视查询词是否被候选覆盖。

### 9.4 混合分数

chunk 和实体的基础混合分数：

~~~text
S_hybrid = alpha × D + (1 - alpha) × L
alpha = 0.72
~~~

默认候选池大小：

~~~text
candidate_limit =
    max(limit, min(200, limit × 8))
~~~

先扩大候选池，再做融合排序，避免 dense 和 lexical 各自只取很小 top-k 造成召回损失。

### 9.5 为什么所有 chunk 都索引

图谱资格与回答价值是两回事：

- 目录可能帮助用户定位章节；
- 参考文献可能回答 “出处是什么”；
- 父级摘要适合宽泛主题查询；
- 页内正文适合精确事实回答。

因此，retrieval-only 只表示 “不能独立生产图事实”，不表示 “没有检索价值”。

---

## 10. 模型调用的上下文安全

### 10.1 Ollama 请求 token 估计

模型请求使用比 chunking 更保守的估计：

~~~text
T_hat_ollama =
    max(1, N_CJK + ceil((N_chars - N_CJK) / 2))
~~~

图片默认额外估算 1024 token/张。

### 10.2 有效上下文

~~~text
effective_context =
    min(
        model_declared_context,
        loaded_runner_context if available
        else configured_fallback
    )
~~~

默认 fallback 为 8192。

请求准入条件：

~~~text
estimated_input
+ max_output
+ safety_margin
<= effective_context
~~~

默认：

- 普通最大输出 2048；
- safety margin 256；
- 固定 Ollama seed 17。

### 10.3 协议级失败

以下情况不能被解释为 “模型没有抽到实体/关系”：

- <code>done != true</code>；
- 结束原因是 length 或 max_tokens；
- 缺失 <code>prompt_eval_count</code>；
- 返回无法修复的非法 JSON；
- 实际输入 token 明显高于估计。

实际输入允许的估计误差：

~~~text
actual_input - estimated_input
<= max(128, ceil(0.50 × estimated_input))
~~~

JSON 解析失败允许一次 temperature=0 的确定性修复重试，仍失败则记录协议错误，不能从任意文本碎片猜测结果。

固定 seed、上下文预算、prompt 版本和语义 contract 都会进入处理指纹。改变这些参数意味着处理 treatment 变化，需要重建相应快照。

---

## 11. 逻辑 Schema：治理语义而不压制召回

### 11.1 两层 Schema

#### 固定物理 Schema

HugeGraph 只使用稳定的 <code>KFDocument</code>、<code>KFChunk</code>、<code>KFEntity</code> 和固定边标签。领域变化不创建新的物理 label。

#### 版本化逻辑 Schema

逻辑 Schema 描述：

- 实体类别；
- 关系名称和自然语言定义；
- domain / range；
- 方向；
- 主体和客体角色；
- 正例、反例和软提示词；
- 结构证据策略；
- 否定、模态和冲突策略；
- 自动发布还是仅审核。

这样既保证图数据库结构稳定，又允许不同项目逐步扩展业务语义。

### 11.2 当前基础实体类型

| 类型 | 含义 |
|---|---|
| <code>GenericEntity</code> | 无法归入更具体类别的命名对象或概念 |
| <code>Person</code> | 自然人、身份或角色 |
| <code>Organization</code> | 组织、机构、企业或团队 |
| <code>Location</code> | 物理地点或行政区域 |
| <code>Product</code> | 产品、设备、组件或材料 |
| <code>Technology</code> | 技术、方法、流程或标准 |
| <code>Event</code> | 完整事件、活动、状态变化或事故 |

### 11.3 当前基础关系

当前基础版本为 <code>general-schema-v13-grounded-open-carrier-v1</code>。

| 关系 | 方向语义 | 发布模式 |
|---|---|---|
| <code>related_to</code> | 保存原文端点顺序的一般语义联系 | 仅审核，不自动发布 |
| <code>source_assertion</code> | 原文实际谓词的完整主体 → 完整客体 | 自动发布开放载体 |
| <code>part_of</code> | 局部/组件 → 整体 | 自动发布 |
| <code>scopes_topic</code> | 行首编号或标签 → 所管辖主题 | 自动发布 |
| <code>belongs_to</code> | 成员/所属项 → 群体或所有者 | 自动发布 |
| <code>located_in</code> | 被定位实体 → 地点 | 自动发布 |
| <code>uses</code> | 主动使用者 → 工具、方法或资源 | 自动发布 |
| <code>produces</code> | 生产者 → 产出物 | 自动发布 |
| <code>depends_on</code> | 依赖方 → 被依赖对象 | 自动发布 |
| <code>causes</code> | 完整原因事件 → 完整结果事件 | 自动发布 |
| <code>supports</code> | 支持提供方 → 被支持功能或目标 | 自动发布 |

<code>related_to</code> 不能作为 “不知道是什么关系时的万能兜底”。这正是避免图中出现大量名为 “关系” 或 “相关” 的边的关键。

### 11.4 开放关系载体

当原文有明确二元谓词，但当前 Schema 没有足够准确的固定关系时，系统可发布：

~~~text
storage_predicate = source_assertion
display_predicate = exact_source_expression
~~~

例如存储类型仍是 <code>source_assertion</code>，界面展示原文中的 “管理”“调度”“兼容”“连接”等实际表达。

开放载体必须满足：

- 来自授权的开放事实清单或原子结构 frame；
- source expression 是原文精确连续文本；
- 主客体和方向明确；
- 非否定；
- 不是单纯共现、同列、相邻或共享第三实体；
- 通过与固定谓词相同的证据和独立验证门槛。

若同一个原文命题已成功绑定到更精确的固定谓词，开放载体会被抑制，避免重复边。

### 11.5 普通构建为何默认不做 Schema 发现

当前默认：

~~~text
SCHEMA_DISCOVERY_DURING_BUILD = false
~~~

原因是 Schema 发现属于治理工作，而不是每次图构建的召回前置条件：

- 大项目每次发现会产生大量模型调用；
- 新谓词审查失败不应阻止原文事实通过开放载体进入图；
- 同一项目构建期间应使用冻结的一致 Schema；
- Schema 变更应可审计、可单独重评。

用户仍可显式触发 “重新评估知识结构”。

### 11.6 显式 Schema 演进

Schema 演进只允许 additive-only：

- 可以新增实体类型和关系类型；
- 不能自动删除、重命名或重新定义活动类型；
- 候选必须有精确原文证据；
- 必须验证语义新颖性、方向、角色、domain/range 和命名质量；
- 最多新增 5 个实体类型、8 个关系类型。

#### 单文档自适应采样

设：

- <code>E</code>：图谱可抽取 chunk 数；
- <code>S</code>：章节数；
- base=12；
- ceiling=48。

~~~text
chunk_target =
    min(
        ceiling,
        max(
            base,
            ceil(1.5 × sqrt(E)),
            ceil(4 × log2(S + 1))
        )
    )
~~~

字符目标：

~~~text
char_target =
    min(
        char_ceiling,
        max(char_base, chunk_target × chars_per_chunk)
    )
~~~

默认：

- <code>char_base=12000</code>
- <code>char_ceiling=60000</code>
- <code>chars_per_chunk=2400</code>
- 最小样本 160 字符
- 每章节最多优先取 2 个
- 模型批次默认 4 个 corpus excerpt

#### 单文档 MMR 选择

~~~text
diversity = 1 - max_cosine(candidate, selected)

score_doc =
    0.55 × diversity
  + 0.30 × information
  + 0.15 × new_section_bonus
~~~

#### 多文档项目采样

设 <code>D</code> 为文档数。每文档候选上限：

~~~text
per_doc_cap =
    ceiling                         if D = 1
    max(2, ceil(ceiling / sqrt(D))) otherwise
~~~

全项目目标：

~~~text
corpus_target =
    min(
        ceiling,
        candidate_count,
        max(
            base,
            min(D, ceiling),
            ceil(1.75 × sqrt(E))
        )
    )
~~~

项目 MMR：

~~~text
score_corpus =
    0.42 × diversity
  + 0.28 × information
  + 0.22 × new_document_bonus
  + 0.08 × new_section_bonus
~~~

#### 新颖性和接受门槛

文档新颖性：

~~~text
document_novelty =
    max_i(1 - nearest_schema_similarity_i)
~~~

默认触发阈值为 0.22。

普通候选接受要求：

~~~text
confidence >= 0.85
AND diverse_evidence_count >= 2
~~~

若只有一条完全冻结、精确可核对的证据，内部可提升到单证据门槛 0.96，但仍需其余政策验证。

---
## 12. 精确来源上下文与实体抽取

### 12.1 为什么先构造 Grounded Context

chunk 是检索和模型预算单位，但不一定是事实最小单位。一个 chunk 可能同时包含：

- 多个句子；
- 一个列表的父项和多个子项；
- 多行表格；
- 标题与正文；
- 跨页续接内容。

如果直接让模型在整个 chunk 上判断关系，模型容易把相邻但无关的实体连在一起。

因此，图谱路径在 chunk 之上再构造 <code>Grounded Context</code>：

- 为每段来源字节建立精确 block-to-chunk map；
- 将正文按句号、分号等切为原子语义范围；
- 列表项、表格行和单元格保持原子；
- 保存标题路径、页码、列表层级和表格角色；
- 后续实体和关系阶段必须使用同一个 context manifest。

当前 facade 是 <code>grounded_context.py</code>，历史持久表和内部版本仍保留 <code>shadow_context*</code> 名称。

关键版本：

- context：<code>shadow-context-source-map-v8-cross-block-lists</code>
- source map：<code>block-sentence-list-or-legacy-local-v8-cross-block-lists</code>
- structure gate：<code>knowledge-forge-structure-gate-v5</code>

### 12.2 可抽取来源角色

允许关系抽取的 parser role：

- <code>body</code>
- <code>list_item</code>
- <code>table_row</code>
- <code>table_cell</code>
- <code>media_description</code>

默认不允许直接产生关系的角色：

- heading；
- page header / footer / number；
- table header / container；
- table of contents；
- index entry；
- caption；
- footnote；
- reference heading。

不允许直接抽取不等于删除。它们仍可作为结构上下文或检索内容。

### 12.3 列表层级

列表父子关系只根据可复现的结构语法恢复：

- 项目符号；
- 编号样式；
- 缩进；
- DOCX 列表级别；
- PDF 中常见符号和私有区 glyph；
- 跨页时允许跳过纯页码或页导航标题。

不会因为两个短段落相邻就猜测父子关系。由此可正确支持：

~~~text
ACoreOSMP653 核心操作系统
  - 分区管理
  - 任务管理
  - 时间管理
  - 定时器管理
~~~

每个子项都可以形成 “子项 part_of 整体” 的候选，但仍需端点、结构和发布校验。

### 12.4 原子范围

正文按句子和分号边界切分。列表项、表格行、单元格和媒体描述按其自然结构保持原子。

一条普通关系证据必须：

~~~text
evidence_span inside one grounded context
AND evidence_span inside one atomic scope
AND evidence covers both endpoint spans
~~~

这条约束直接消除大量跨句共现假关系。

### 12.5 Context 身份

context ID 和 manifest 会包含：

- 内容修订；
- context kind；
- source map 版本；
- parser role；
- 来源字节或来源哈希；
- 结构策略版本。

实体运行与关系运行若使用不同 context manifest，关系阶段拒绝复用实体端点，避免 “实体来自旧分块，关系来自新分块” 的静默错配。

---

## 13. 实体 Mention 清单与精确落点

### 13.1 实体阶段的目标

实体阶段不直接回答 “哪些 mention 是同一个现实对象”。它先完成两个更基础的任务：

1. 找全可以充当关系端点的、具有指称意义的 mention。
2. 把每个 mention 精确锚定到原文。

归一化留给下一阶段。

### 13.2 不可变实体命名空间

当前实体处理版本：

<code>knowledge-forge-grounded-entity-v40-soft-wrap-suffix-recovery</code>

实体 namespace 由快照和完整输入指纹确定：

~~~text
namespace_id =
    "grounded_entity_"
    + SHA256(snapshot_id_or_live + "|" + input_fingerprint)
~~~

指纹包含：

- 项目和策略；
- Schema 版本及哈希；
- 文档、活动修订和内容哈希；
- chunks、chunk 哈希和抽取资格；
- grounded context manifest；
- LLM、embedding 和 reranker treatment；
- 固定 seed；
- token 预算；
- 实体策略、配置和代码版本。

历史数据库中部分实际 ID 仍可能以 <code>shadow_entity_</code> 开头，这是稳定存储兼容，不影响当前策略语义。

### 13.3 持久 work item

每个 chunk 的实体工作项拥有稳定身份：

~~~text
work_item_id =
    SHA256(
        namespace_id,
        chunk_id,
        revision_id,
        chunk_content_hash
    )
~~~

work item 可处于 pending、leased、retryable_failed、terminal_failed 或 completed。服务重启后可继续 claim，而不是重跑整个项目。

### 13.4 模型实体清单

当前默认配置：

| 参数 | 默认值 |
|---|---:|
| 来源约束 inventory | 开启 |
| 单段最大字符 | 2400 |
| 单段最大句数 | 8 |
| 单段最大返回项 | 96 |
| gleaning 轮数 | 1 |
| temperature | 0 |

模型任务被刻意限制为：

- 列出文本中有明确指称的实体 mention；
- 返回原文复制的 surface；
- 返回证据；
- 给出逻辑 Schema 类型提示；
- 不在这一阶段生成关系。

把实体和关系拆开，可以避免模型为了输出一个三元组而遗漏没有立即成边的实体。

### 13.5 递归拆分

如果输入超过 2400 字符，或模型输出达到 item 上限、发生截断，系统在句子边界附近递归拆分，最大深度 10。

其目的不是无限重试，而是将 “响应容量不足” 转换为更小、可验证的确定性子任务。

### 13.6 Gleaning

第一轮清单完成后，第二轮不从头再问，而是携带：

- 第一轮已经接受的 mention；
- 明显失败项；
- 尚未覆盖的句子或结构。

模型只补遗漏。默认最多 1 轮，防止长文档调用次数失控。

### 13.7 确定性召回补充

模型确认一个精确实体 surface 后，系统会在同一来源范围中搜索其其他精确出现，恢复漏掉的重复 mention。

恢复时考虑：

- ASCII 大小写和标识符边界；
- Unicode 归一化；
- 嵌套 mention；
- 短词误包含；
- 软换行。

另有通用结构端点候选：

| 来源 | 初始候选强度 |
|---|---:|
| policy cue endpoint | 0.55 |
| <code>A --rel--&gt; B</code> 结构端点 | 0.80 |
| 列表项 | 0.75 |
| section / inline label | 0.70 / 0.72 |

这些分数只用于端点召回和排序，不能让候选绕过 mention 校验，更不能直接发布关系。

### 13.8 Span sanitizer

一个 mention 必须满足：

~~~text
content[start:end] == surface
~~~

允许的修复仅包括：

- 原偏移精确时去除边缘标点；
- Unicode/casefold 标识符归一；
- 解析器已经确认的 PDF/DOCX 软换行。

默认拒绝：

- 纯数字、年份、日期、百分比、电话、金额、单位；
- 页码、标题编号等结构节点；
- 代词和指示词；
- 只有修饰语的片段；
- 不完整从句；
- 截断的词边界；
- 把完整事件中的内部普通名词错误当成事件；
- 不可拆分的复合端点。

这一步是图谱精度的第一道硬门槛。

### 13.9 Mention 和 occurrence 身份

~~~text
mention_id =
    SHA256(
        project_id,
        document_id,
        revision_id,
        chunk_id,
        start,
        end,
        surface
    )
~~~

~~~text
occurrence_id =
    SHA256(
        document_id,
        source_anchor,
        source_coordinates,
        surface
    )
~~~

- mention 表示抽取和审核视角下的实体提及。
- occurrence 表示在不可变来源中的具体出现。

身份判断只读取以该 occurrence 为中心的一句上下文半径，最多 1200 字符，避免用整篇文档中的同名信息污染局部身份。

---

## 14. 实体归一与保守消歧

### 14.1 为什么不能只用字符串或向量

以下名称相似，但不一定是同一实体：

- ACoreOS 与 ACoreOSMP653；
- task 与 Task_ID；
- “系统” 与 “系统调用”；
- v1.2 与 v2.1；
- 中文两字词的一字差异。

以下名称差异很大，却可能相同：

- 全称与缩写；
- 中文名与英文名；
- 大小写或下划线差异；
- 公司名与官方简称。

所以归一化是多阶段决策，而不是一个相似度阈值。

### 14.2 候选召回

对每个新 mention，先召回：

- 精确 canonical name / alias；
- 词法候选，默认最多 10；
- 向量候选，默认最多 20；
- 合并后最多 30 个进入 reranker；
- 最多 5 个进入 LLM 身份复核。

### 14.3 硬冲突

在打分前先排除 cannot-link：

- 明确不同数字标识或版本；
- 短中文词的近编辑距离歧义；
- 已知且不兼容的具体类型；
- 来源中明确冲突的身份属性。

<code>GenericEntity</code> 与具体类型之间不是硬冲突，因为 Generic 往往只是信息不足。

### 14.4 名称特征

词法特征综合：

- 完全相等；
- 规范化后相等；
- 包含关系；
- SequenceMatcher；
- 字符 n-gram Jaccard；
- 首字母缩写；
- 保守单复数等价。

取其中最可信特征作为 <code>lexical</code>。

### 14.5 初排分数

~~~text
C0 =
    min(
        1,
        0.52 × lexical
      + 0.38 × vector
      + 0.10 × type
    )
~~~

### 14.6 Reranker 分数

~~~text
C1 =
    min(
        1,
        0.34 × lexical
      + 0.30 × vector
      + 0.26 × reranker
      + 0.10 × type
    )
~~~

Reranker 读取两个 mention 的名称、类型和局部身份证据，比单纯 embedding 更能识别简称与复合名称。

### 14.7 合并决策

决策顺序：

1. 精确 alias/name 且无硬冲突：确定性合并。
2. 唯一且保守的单复数等价：可合并。
3. 非精确候选：即使 <code>C1</code> 达到自动阈值，也不直接合并，进入 bounded identity review。
4. 复核只看最多 5 个候选。
5. 复核分数低于 0.68、模型无效、结果不确定或调用失败：创建新 canonical。

代码中保留 0.90/0.93 等自动候选阈值用于排序和触发复核，但当前保守策略不允许普通非精确候选只凭高分自动合并。

### 14.8 嵌套名称策略

共享中心词的嵌套短语通常保持分离，例如：

~~~text
任务
周期任务
周期任务管理
~~~

除非有明确简称形态或局部身份支持，否则不会因为包含关系合并。

### 14.9 Canonical 实体

新 canonical key 由 namespace、归一策略和 mention 身份哈希生成，避免同名跨项目或跨快照碰撞。

实体保留：

- canonical name；
- aliases；
- entity type；
- 所有 mention；
- 文档、chunk 和 source span；
- 归一候选分数；
- 复核请求与响应；
- cannot-link 原因。

### 14.10 压缩率诊断

~~~text
compression =
    max(0, 1 - canonical_count / mention_count)
~~~

这个值只用于诊断：

- 太低可能表示别名没有合并；
- 太高可能表示过度合并；
- 不同领域的合理值差异很大，因此没有跨项目统一的 “越高越好”。

---

## 15. 关系抽取、验证与发布

### 15.1 关系处理的三层模型

这是整个策略最关键的设计。

| 层 | 允许什么 | 不代表什么 |
|---|---|---|
| 候选 candidate | 模型开放事实、结构投影、Schema shortlist、pair cue | 不代表关系真实 |
| 证据 evidence | 精确来源范围、端点区间、原文谓词、方向审核 | 不代表可自动发布 |
| 事实 fact | 独立验证通过、归一化、无冲突、新鲜、谓词可发布 | 才能进入正式图 |

所以 “模型给了 0.9” 不能直接写边；“没有进入图” 也不等于模型没发现，它可能是 UNCERTAIN、冲突、技术失败或 review-only。

当前关系语义 contract：

<code>grounded-relation-v2-semantic-contract-2026-08-12</code>

### 15.2 当前默认策略

| 配置 | 默认 |
|---|---:|
| source-first open facts | 开启 |
| source inventory gleaning | 1 轮 |
| Schema binding gleaning | 0 轮 |
| semantic shortlist | 每 frame 1 个 |
| shortlist 最低分 | 0.18 |
| 每 context 开放载体上限 | 96 |
| reference adjudication | 开启 |
| verifier 模式 | lean |
| verifier batch | 8 |
| verifier 最大字符 | 48000 |
| focused review 上限 | 24 |
| 单请求最大协议尝试 | 3 |
| 二次原因分流 | 已实现，默认关闭 |
| pair 句距窗口 | 1 |
| pair 字符距离 | 384 |
| 每 chunk pair 预算 | 256 |
| 旧流程 fallback | 禁止 |

### 15.3 Pair opportunity ledger

一个原子范围有 <code>n</code> 个不同 occurrence 时，理论无序端点对数量：

~~~text
P = n(n - 1) / 2
~~~

系统为每个机会建立不可变 ledger，即使它后来被规则拒绝或预算延迟。这样可以区分：

- 模型没看见；
- 规则明确拒绝；
- 达到预算尚未处理；
- 技术失败；
- 审核不通过。

~~~text
pair_identity =
    SHA256(revision_id, left_occurrence_id, right_occurrence_id)
~~~

opportunity identity 还会加入关系运行和策略版本。

### 15.4 Pair 预过滤

在调用模型前排除：

- 同一 canonical 的非法自环；
- 不同表格数据行的普通邻接；
- 已知且不兼容的类型；
- 不合格来源角色；
- 不允许的结构或 atomic scope；
- 句距和字符距离过远。

字符中心距离：

~~~text
d_char =
    abs(
        floor((s1 + e1) / 2)
      - floor((s2 + e2) / 2)
    )
~~~

以下结构可绕过普通距离门槛：

- 同一单元格；
- 同一表格行；
- 同一原子范围；
- 直接列表父子；
- 直接 section-label 父子。

### 15.5 Pair 优先级

~~~text
priority = (
    3,
    structure_rank,
    type_rank,
    cue_rank,
    sentence_distance,
    char_distance,
    left_id,
    right_id
)
~~~

排序稳定，因此同一快照重跑时优先顺序可复现。

超过每 chunk 256 的 pair 不会静默删除，而是标记 <code>DEFERRED_BUDGET</code>。这对质量审计很重要：预算不足不应伪装成负样本。

### 15.6 Source-first 开放事实清单

当前主路径不是先对所有 pair 问 “是否有 Schema 关系”，而是先按来源范围让模型枚举原文直接陈述的二元事实。

输入包括：

- 冻结的 occurrence ID；
- 原文和精确坐标；
- 角色、并列、列表、表格与层级信息；
- 第一轮已接受和失败项；
- 覆盖检查表。

模型返回：

- subject occurrence；
- object occurrence；
- 原文实际谓词；
- 连续 evidence；
- 方向；
- 否定和模态信息。

这样先恢复 “文档说了什么”，再决定 “Schema 里叫什么”，比先给固定谓词列表更有召回能力。

### 15.7 开放事实严格解析

每个模型事实必须通过：

- occurrence ID 存在且属于当前 context；
- subject 与 object 不相同且不重叠；
- evidence 是原文连续子串；
- evidence 同时覆盖两个端点；
- 原文谓词是有内容的关系头，不只是 “和、与、可能、可以”；
- evidence 位于一个 atomic scope；
- 方向可由原文角色解释。

非法项不会用模糊字符串匹配 “尽量修好”，而是进入失败审计或下一轮 omission prompt。

### 15.8 Gleaning 和引用消解

默认有一轮遗漏补充。第二轮只问：

- 哪些 occurrence 尚未出现在已接受事实中；
- 哪些原子范围没有被覆盖；
- 第一轮哪些输出因格式或边界失败；
- 是否遗漏了并列主语、并列宾语或列表子项。

代词或指代表达进入独立、受限的 reference adjudicator。只有能唯一映射到精确 antecedent 才允许替换，不能凭整篇文档常识猜测。

### 15.9 通用结构投影

除模型清单外，系统还从解析结构生成候选 frame：

- 括号枚举；
- 表头到当前行值的角色绑定；
- 行首标签或编号到条款主题；
- 直接列表父子；
- section label 到子内容；
- 并列主语或宾语；
- 所有格；
- 多角色连接表达；
- 显式箭头或结构谓词。

这些规则只生成候选，不直接发布。因此它们可以提高截图式列表、表格和技术说明书的召回，而不会把所有列表相邻项互相连边。

### 15.10 Schema 绑定

每个 source frame 与逻辑关系定义、正反例、端点示例、角色政策和类型约束计算 shortlist。

文本表示由：

- Unicode word；
- 紧凑字符 trigram；
- 定义和例句；
- source expression；
- endpoint surface；
- 结构角色；

构成，语义相似度使用 Jaccard 型重叠。

综合分数：

~~~text
R =
    0.36 × relation_similarity
  + 0.25 × evidence_example_similarity
  + 0.22 × endpoint_example_similarity
  + 0.12 × role_policy_similarity
  + 0.05 × type_score
~~~

默认 shortlist 阈值为 0.18，每个 frame 保留 top 1。对没有编译结构契约的一般词法二元 frame，门槛提高到：

~~~text
max(0.18, 0.34) = 0.34
~~~

低门槛只是为了候选召回，后面还有硬验证；它不是发布阈值。

### 15.11 开放载体补召回

若没有固定 Schema 谓词准确覆盖一个精确 source frame，可生成 <code>source_assertion</code> 候选，并保存规范化 source expression。

它解决的是：

- Schema 不可能预先枚举所有领域动词；
- 强行归入 <code>uses</code>、<code>supports</code> 等会损害精度；
- 直接丢弃又损害召回。

开放载体不是 <code>related_to</code>。前者保留原文具体谓词、精确方向和证据；后者只是通用语义联系且当前仅供审核。

### 15.12 候选验证

进入独立验证器前，候选必须满足：

1. 运行、pair、context 和端点身份与不可变输入一致。
2. evidence 位于 context 内。
3. evidence 覆盖两个精确端点。
4. predicate 存在于活动 Schema。
5. parser role 和结构类型被该关系政策允许。
6. 列表或 section 的 child → parent 方向正确。
7. 开放载体来源被授权，source expression 和 span 精确。
8. evidence 不跨 atomic scope。
9. 不存在非法自环。
10. domain/range 和 inverse metadata 合法。

类型仍不确定的候选可以进入验证器补充角色判断；<code>review_only</code> 关系最多进入 UNCERTAIN，不能自动发布。

### 15.13 独立有序蕴含验证

普通候选由独立 final verifier 判断，验证器与候选生成调用分离。

SUPPORTED 至少要求：

~~~text
exact_evidence = true
subject_role_supported = true
object_role_supported = true
ordered_entailment_fields_complete = true
~~~

对非对称关系：

~~~text
SUPPORTED iff
    (
        forward  = ENTAILED
        AND reverse = NOT_ENTAILED
    )
    XOR
    (
        forward  = NOT_ENTAILED
        AND reverse = ENTAILED
    )
~~~

如果只有反向被支持，解析器可以在所有原文端点仍精确的前提下交换端点，并记录 <code>direction_normalization</code>。这不是静默猜反向，而是将验证器明确确认的方向归一化。

对对称关系：

~~~text
SUPPORTED iff
    forward = ENTAILED
    AND reverse = ENTAILED
~~~

验证器即使返回 “SUPPORTED”，若缺失精确证据、角色布尔值或有序蕴含字段，也会被确定性降为 REJECTED。

模型超时、截断、JSON 协议错误属于 retryable failure，最多 3 次，不会被当作 REJECTED。

### 15.14 结构契约的验证

列表父子、表格行角色等 parser contract 不必再次让 LLM猜测结构是否存在，而是从不可变 parser metadata 重新计算：

- 确实是直接父子；
- 缩进和 marker 级别匹配；
- 表头和当前行属于同一表；
- 标签直接管辖当前条款；
- 方向满足 Schema。

这种 “确定性结构重验 + 模型语义验证” 比让模型重复识别版式更稳定。

### 15.15 Focused review

批量 verifier 后，可对最多 24 个高风险接受或拒绝项做聚焦复查，例如：

- 方向边界；
- 类型刚好不确定；
- 开放 source expression；
- 角色可能倒置；
- 复杂并列；
- 证据覆盖临界。

聚焦复查仍不能绕过最终发布门槛。

### 15.16 二次原因分流

代码中实现了以下二次路线：

- <code>EVIDENCE_RECOVERY</code>：端点正确但 evidence span 不完整；
- <code>POLARITY_REVIEW</code>：否定或模态；
- <code>STRUCTURAL_REVIEW</code>：列表、表格或标签结构；
- <code>PREDICATE_REFINEMENT</code>：<code>related_to</code> 或谓词过粗；
- <code>TYPE_ROLE_REVIEW</code>：端点类型或角色；
- <code>ROLE_RETRY</code>：独立重问主客体角色；
- <code>TECHNICAL_RETRY</code>：协议、截断或超时。

以下原因直接不进入二次召回：

- 明确只是共现；
- 跨原子范围；
- 明确类型不兼容；
- 仅列表/表格同级相邻；
- 端点并非完整实体；
- 原文明确否定且无可发布命题。

当前默认 <code>RELATION_SECONDARY_ENABLED=false</code>。原因是 source-first、结构投影和独立 verifier 已成为主路径；二次分流虽然具备能力，但在全面评估调用成本和净增益前不应默认扩大处理时间。

如果未来启用，恢复结果仍必须再次走普通 validation 和新的独立 verifier，不能让 “二次流程” 成为降低标准的旁路。

### 15.17 关系状态解释

| 状态 | 含义 | 是否进入图 |
|---|---|---:|
| SUPPORTED / publishable | 独立证据、角色和方向均通过 | 是 |
| UNCERTAIN | 有候选价值，但发布条件不完整或谓词仅审核 | 否 |
| REJECTED | 语义、结构、端点、方向或证据明确不成立 | 否 |
| RETRYABLE_* | 超时、截断、协议或模型技术失败 | 否，等待重试 |
| TERMINAL_FAILED | 超过重试上限 | 否，保留审计 |
| DEFERRED_BUDGET | 在机会账本中，但本轮预算未处理 | 否 |

<code>UNCERTAIN</code> 的作用不是在图中显示一条低置信边，而是保留候选、原因、证据和复核路径，供后续策略改进或人工审计。

### 15.18 关系归一

只接受每个 hypothesis 的第一个独立 SUPPORTED attempt。

方向归一：

- 正向关系：保持；
- inverse：交换端点并换成 inverse predicate；
- symmetric：按稳定 canonical key 排序端点。

语义事实键：

~~~text
fact_key =
    "relfact_"
    + SHA256(
        project_id,
        subject_canonical,
        predicate,
        object_canonical,
        normalized_source_expression_if_open
    )
~~~

证据键：

~~~text
evidence_key =
    SHA256(
        fact_key,
        revision_id,
        subject_occurrence,
        object_occurrence,
        predicate,
        evidence_spans,
        hypothesis_id
    )
~~~

归一化会：

- 合并完全重复事实；
- 合并对称或 inverse 的重复表示；
- 保留同一端点之间合法的不同谓词；
- 拒绝非法自环；
- 将同一非互惠谓词的双向独立支持标记为 reciprocal conflict，并暂不发布两边；
- 精确 fixed predicate 已覆盖时抑制开放载体重复。

### 15.19 最终物化门槛

一条关系进入正式图必须同时满足：

~~~text
fresh_input_fingerprint
AND complete_relation_snapshot
AND current_entity_assignments
AND current_active_schema
AND exact_endpoint_spans
AND exact_supported_evidence
AND legal_direction
AND legal_domain_range
AND publish_mode = auto
AND no_unresolved_conflict
AND legal_self_loop_policy
AND independent_verifier_supported
~~~

这就是项目最重要的质量保证：候选生成负责召回，最终发布不依赖单一模型的自报置信度。

### 15.20 “置信度都是 1” 的解释

HugeGraph 物理兼容字段 <code>kf_confidence</code> 可能统一为 1.0。它不代表统计校准后的 100% 真实概率。

当前正式事实的权威质量信号是：

- publication state；
- independent verifier verdict；
- evidence count；
- 原文证据；
- source document / chunk / revision；
- conflict state；
- policy/schema version。

因此，界面不应把统一的 1.0 当作有区分度的 “关系置信度” 展示。

---

## 16. 图谱生成、发布和 HugeGraph 物化

### 16.1 项目级构建顺序

<code>_ingest_knowledge_forge_strategy</code> 的主流程是：

1. 准备或复用目标文档内容修订。
2. 读取项目中所有已索引、未删除的活动文档。
3. 冻结本次项目级 Schema。
4. 构建或复用不可变实体 namespace。
5. 构建关系 run。
6. 生成实体、关系和社区展示描述。
7. 开启新的 strategy graph generation。
8. 计算完整 graph policy hash。
9. 获取图写入 lease。
10. 将当前 generation 全量同步到 HugeGraph 隔离 namespace。
11. 执行社区检测。
12. 用 CAS 激活 ready generation。
13. 将文档策略状态和 durable job 标记完成。

### 16.2 图策略哈希

图 policy hash 覆盖：

- strategy；
- Schema 版本和哈希；
- Schema discovery treatment；
- 实体解析策略和 entity namespace；
- 关系策略、semantic contract 和 relation run；
- 完整输入 fingerprint；
- 图物化配置；
- 社区算法和产物版本。

只要任一重要输入变化，旧图就不能冒充新鲜结果。

### 16.3 Lease 与 CAS

默认图写 lease 为 900 秒。写入上下文包含：

~~~text
GraphMutationContext(
    namespace,
    generation,
    policy_hash,
    lease_token
)
~~~

所有图修改必须验证：

- lease token 仍属于当前 worker；
- generation 未被更新任务取代；
- policy hash 未变化；
- 用户未请求取消。

完成时：

~~~text
UPDATE active_graph
SET ready_generation = new_generation
WHERE current_generation = expected_generation
  AND lease_token = expected_token
  AND policy_hash = expected_hash
~~~

受影响行数为 0 表示 CAS 失败，当前 worker 已过期，不得激活其结果。

### 16.4 为什么采用全量 generation 替换

Knowledge Forge 在内容贡献层支持文档级替换，但图发布采用完整项目快照：

- 实体归一可能因新增文档发生变化；
- 一条事实可能有多个文档证据；
- 社区划分是全图函数；
- 增量修改容易留下旧别名、旧边或旧社区。

全量 generation 在 HugeGraph 侧更容易保证读者只看到一个一致版本。解析、embedding、entity namespace 和部分 relation artifacts 仍尽可能复用，控制重建成本。

### 16.5 固定物理图 Schema

#### 顶点

| Label | 含义 |
|---|---|
| <code>KFDocument</code> | 文档 |
| <code>KFChunk</code> | 内容块 |
| <code>KFEntity</code> | canonical 实体 |

#### 边

| Label | 方向 |
|---|---|
| <code>kf_contains</code> | 文档 → chunk |
| <code>kf_mentions</code> | chunk → entity |
| <code>kf_related</code> | entity → entity |

图记录按情况携带：

- <code>project_id</code>
- <code>project_key</code>
- strategy namespace
- generation
- policy hash
- document / chunk / assertion ID
- relation type
- evidence 和描述
- Schema version
- update time

顶点和边 ID 使用项目、namespace 和稳定业务 key 生成，避免跨项目串图。

### 16.6 PostgreSQL 才是事实审计权威

HugeGraph 面向：

- 可视化；
- 邻居展开；
- 社区浏览；
- 图遍历。

PostgreSQL 面向：

- 不可变实体与关系运行；
- 验证尝试；
- 证据区间；
- 归一事实；
- 冲突和折叠审计；
- 新鲜度；
- RAG 排名快照。

因此，HugeGraph 丢失或需要重建时，可以从 PostgreSQL 的已验证快照重新物化，而不是重新调用所有模型。

---

## 17. 实体、关系和社区描述

### 17.1 描述是展示适配器

描述版本：

<code>knowledge-forge-grounded-description-v1</code>

描述生成发生在事实已经确定之后，它不会改变：

- 实体抽取；
- 实体归一；
- 关系方向；
- verifier verdict；
- 发布状态。

这样可以单独改进用户可读性，而不污染事实真值。

### 17.2 实体描述

实体描述由以下内容确定性组合：

- 实体名称和类型；
- 最多 2 条不同来源的精确证据片段；
- 最多 4 条已验证相邻关系；
- aliases 和来源数量。

它回答 “这个实体在项目文档中是什么”，而不是让模型基于常识自由扩写。

### 17.3 关系描述

关系描述包含：

- subject；
- display predicate；
- object；
- 精确来源证据；
- 文档和 chunk；
- 方向。

对于 <code>source_assertion</code>，展示的是 source expression，不向用户暴露笼统存储名。

### 17.4 社区检测

Knowledge Forge 只使用 GraphRAG 的 BYOG/Leiden 社区检测能力，不使用 GraphRAG 的实体关系抽取。

社区输入来自当前已验证图：

1. 排除自环。
2. 同一无序实体对上的多条有向事实合成一条用于聚类的无向边。
3. 边权重为这对实体之间已验证关系条数：

~~~text
community_edge_weight =
    count(verified_relation_rows_for_unordered_pair)
~~~

4. 不使用统一的 confidence 字段作为边权。
5. 未连接实体保留为 singleton community。

默认：

| 参数 | 值 |
|---|---:|
| 随机 seed | 42 |
| max cluster size | 10 |
| use LCC only | false |
| timeout | 300 秒 |

社区输入哈希包含项目、generation、entity namespace、配置、canonical 集和关系集。

稳定社区 ID：

~~~text
community_id =
    SHA256(
        project_id,
        community_input_hash,
        level,
        sorted_member_ids
    )
~~~

### 17.5 社区描述

社区描述不是简单输出 “包含 178 个实体、250 条关系”。当前确定性 narrative 会：

1. 找 focal entity；
2. 提取其主要一阶邻居；
3. 汇总社区最常见的关系语义；
4. 选择覆盖代表实体和焦点关系的最佳原文片段；
5. 生成用户可读的 “该社区主要介绍什么”。

焦点实体按以下优先级排序：

~~~text
(
    unique_neighbor_count DESC,
    incident_relation_count DESC,
    source_count DESC,
    entity_name ASC,
    entity_key ASC
)
~~~

当前 narrative 版本为 <code>verified-graph-theme-narrative-v2</code>。

因为摘要只使用已验证图和原文证据，所以它不会在社区描述中凭空补充项目外知识。

### 17.6 社区描述在哪里展示

当前界面主要在两个位置展示：

- 全局社区地图右侧的社区目录卡片中，以截断摘要帮助用户判断社区主题。
- 点击进入某一社区后，在右侧社区详情中展示更完整的社区说明、代表实体和主要关系。

---

## 18. 混合 RAG：文本、实体与图关系协同

### 18.1 查询前的新鲜度门槛

图检索只读取：

- <code>status=ready</code>；
- snapshot complete；
- 当前源内容仍匹配；
- 当前策略、Schema 和 code treatment 匹配；

的快照。

若图 stale 或不完整，系统不能把旧关系包装成当前答案。文本知识库仍可根据其自身活动内容修订工作。

### 18.2 三类候选

#### 文本块

由 dense + lexical 混合检索产生。

#### 实体

由实体名称、alias、描述的 lexical + vector 联合召回。

若规范化实体名或 alias 长度至少 2 且完整包含在 query 中，将其标记为 query surface match，并使 seed score 至少为 1。

#### 图关系

从最多 10 个高质量 seed 出发：

1. 查询 seed 直接关系和命中 chunk 支持的关系。
2. 根据 distinct seed links 和 lexical 选最多 24 个一跳邻居。
3. 再查询这些邻居的关系。
4. 总路径最多两跳。

### 18.3 关系语义分数

分别比较 query 与：

- “subject + relation + object” 整体表面；
- predicate label。

~~~text
S_relation_semantic =
    0.45 × cosine(query, endpoint_relation_surface)
  + 0.55 × cosine(query, predicate_label)
~~~

结果截断到 0 到 1。

### 18.4 文本证据预分

~~~text
S_text =
    0.72 × supporting_chunk_score
  + 0.28 × lexical
~~~

### 18.5 图关系最终分数

~~~text
G =
    0.22 × network
  + 0.12 × seed_relevance
  + 0.36 × relation_semantic
  + 0.15 × relation_surface_lexical
  + 0.08 × text_score
  + 0.02 × lexical
  + 0.05 × shared_seed_score
~~~

其中 network：

| 候选来源 | network |
|---|---:|
| seed 网络内部 | 1.00 |
| seed 边界 | 0.88 |
| 扩展邻居 | 0.66 |
| 仅文本证据、无图连接 | 0 |

共享 seed：

~~~text
shared_seed_score =
    min(1, distinct_seed_links / 3)
~~~

### 18.6 事实级去重

检索多样性签名由：

- 无序端点；
- 关系语义；
- evidence；

组成。一个事实有多条来源证据时，先返回 primary，再按需要保留 alternate，避免 top-k 被同一事实的重复证据占满。

最终结果上限为 50。

### 18.7 为什么图检索不直接依赖 HugeGraph 排名

当前排名读取 PostgreSQL 的不可变事实快照，原因是：

- 可同时获得精确 evidence 和验证状态；
- 容易做新鲜度检查；
- 排名公式可测试；
- HugeGraph 物化失败时不会把半图当真值；
- 同一事实多证据的去重更清晰。

HugeGraph 主要服务交互式浏览和拓扑遍历。

### 18.8 图谱在 RAG 中的定位

图谱擅长：

- 连接不同 chunk 中的同一实体；
- 回答 “谁依赖谁、谁组成谁、什么导致什么”；
- 找到一跳、两跳相关事实；
- 提供可审计的短证据链。

文本知识库擅长：

- 保留完整限定条件；
- 返回段落、表格和说明；
- 回答没有适合图结构的问题；
- 弥补图抽取的合理召回损失。

因此系统不追求把文档每句话都变成图。合理目标是：高价值实体关系达到可接受召回，错误边严格受控，并由原文检索覆盖剩余信息。

---

## 19. 图谱浏览与前端可视化策略

### 19.1 技术栈

当前图可视化使用：

<code>@cosmos.gl/graph 3.4.0</code>

它是 WebGL 图渲染方案。前端 bundle 由 esbuild 构建，没有额外引入 ECharts、AntV G6、Cytoscape.js、D3.js 或 vis-network。

### 19.2 有界场景

浏览器不会一次绘制无上限全图，而是提供不同有界视图：

| 场景 | 节点上限 | 边上限 |
|---|---:|---:|
| 全局社区概览 | 42 | 跨社区主边 32 |
| 关系场景 | 48 | 60 |
| 社区重点关系 | 120 | 190 |
| 社区更多关系 | 320 | 560 |
| 社区全部关系 | 2000 | 4000 |
| 实体邻域 | 120 | 160 |

有界展示只影响视图，不会删除数据库事实。

### 19.3 全局社区图

全局社区图中：

- 一个节点代表一个社区；
- 节点大小反映社区实体规模或连接重要性；
- 两个社区之间的双向或多条事实在概览层合并为一条无向聚合边；
- 聚合边权是跨社区已验证事实数量；
- “主要关系” 只显示骨干跨社区连接；
- “全部关系” 显示当前概览预算内的全部聚合连接。

社区之间看见一条线，不意味着底层只有一条实体关系。点击聚合关系后可查看其关系数量和组成。

### 19.4 社区图与实体图

进入社区后：

- 节点代表实体；
- 边代表已验证关系；
- 有向关系保留箭头；
- 点击节点显示实体名称、类型、别名、描述、来源和相邻事实；
- 点击边显示 subject、实际谓词、object 和原文证据；
- 右侧不再保留与当前上下文无关的全局社区目录。

“重点关系 / 更多关系 / 全部关系” 是互斥视图模式：

- 重点关系：展示社区中最能说明主题的核心实体和关系。
- 更多关系：扩大到较多实体和事实，适合继续探索。
- 全部关系：在安全上限内加载该社区完整事实，适合细查。

### 19.5 子图选择

实体子集先选择高连接、高来源覆盖的 hub，再沿真实拓扑扩展。

单节点扩展 fan-out：

~~~text
fanout_cap =
    max(5, min(14, sqrt(edge_limit)))
~~~

这样保留自然网络结构，而不是把实体均匀排成圆。

### 19.6 节点大小连续映射

设节点 importance 为 degree、成员数或场景已有重要性指标：

~~~text
z = log(1 + importance)
lower = percentile_10(z)
upper = percentile_90(z)
u = clip((z - lower) / (upper - lower), 0, 1)

node_size =
    min_size
    + (max_size - min_size) × u^0.76
~~~

范围：

- 社区概览约 18–42 px；
- 实体场景约 16–38 px。

对数和分位数共同抑制极端 hub，避免一个超大节点压住全图。

### 19.7 边样式

默认边宽：

~~~text
edge_width =
    clip(
        1 + 0.09 × log(1 + weight),
        1.0,
        1.3
    )
~~~

默认：

- 低饱和蓝灰；
- opacity 约 0.32；
- 轻微曲率；
- 有向箭头保留。

hover 或 selected 时：

- 直接相邻边约 2.3 px、opacity 约 0.94；
- 无关边 opacity 降到约 0.09；
- 当前节点橙色；
- 一阶邻居蓝色高亮；
- 无关节点降低透明度但不完全消失。

### 19.8 确定性力导向布局

布局先在应用层做有界确定性 relaxation，然后关闭运行时 simulation，避免每次加载抖动、旧图残影和用户操作后重新排布。

迭代次数按节点规模自适应：

| 节点数 | 迭代 |
|---:|---:|
| ≤ 50 | 220 |
| ≤ 120 | 165 |
| ≤ 260 | 90 |
| 更大 | 28 |

斥力近似：

~~~text
repulsion =
    R / (distance^2 + 2600)
~~~

其中：

- compact scene：<code>R=104000</code>
- overview：<code>R=148000</code>

碰撞最小距离：

~~~text
min_distance =
    radius_a / 2
    + radius_b / 2
    + padding
~~~

- compact padding 约 30；
- overview padding 约 40。

重叠时的额外推力：

~~~text
collision =
    (min_distance - distance) × 1.16
~~~

弹簧目标长度：

~~~text
target_length =
    ideal_length + radius_a + radius_b
~~~

- compact ideal length 约 196；
- overview ideal length 约 286。

弹簧力：

~~~text
spring =
    (distance - target_length) × 0.029
~~~

中心拉力：

~~~text
center_force_x = -0.0026 × x
center_force_y = -0.0026 × y
~~~

单步最大位移从约 20 逐渐退火到 5。

### 19.9 标签

- 核心节点默认显示标签。
- 普通节点在 hover、selected、成为一阶邻居或放大到足够程度时显示完整标签。
- 相对缩放达到约 2.2 后，允许显示更多小节点标签，候选上限约 180。
- 标签约 11–12 px，深蓝灰文字和半透明白底。
- 使用碰撞检测避免标签与节点、标签与标签严重重叠。
- 数据名称不被改写，例如 “社区 19” 必须完整保留。

### 19.10 交互

- 普通滚轮：页面上下滚动。
- Ctrl + 滚轮：缩放图谱。
- 按住拖动：平移画布，鼠标显示抓手。
- 点击节点：固定当前节点及一阶邻居高亮，并展示详情。
- 点击边：固定关系高亮并展示证据。
- 点击空白：清除固定高亮。
- Ctrl + 点击拖拽：只进行视图操作，不误清除 selection。
- 前进、后退：在图浏览历史中移动。
- Home：返回全局社区图。
- 适应画布：重新将当前图 fit 到可视区域。

加载状态直接覆盖在图画布中，并在新数据到达前清理旧 renderer 状态，避免上一张图残影。

---

## 20. PostgreSQL 数据模型与审计链

### 20.1 数据分层

数据库不是只保存最终节点和边，而是保留完整漏斗：

~~~text
source document
  -> content revision
  -> section / chunk
  -> grounded context / source map
  -> entity mention / occurrence
  -> canonical assignment
  -> relation pair opportunity
  -> source/model proposal
  -> hypothesis
  -> validation
  -> verification attempt
  -> normalized fact
  -> fact evidence / conflict
  -> graph generation
~~~

### 20.2 主要表族

| 表族 | 代表内容 |
|---|---|
| 项目与模型 | <code>projects</code>、<code>models</code>、<code>settings</code> |
| 文档内容 | <code>documents</code>、<code>document_content_revisions</code>、<code>sections</code>、<code>chunks</code> |
| 检索 | <code>vector_embeddings</code>、索引失败记录 |
| 任务 | <code>jobs</code>、工作项、lease、operation metrics |
| Schema | <code>graph_schemas</code>、Schema evaluation run、policy snapshot |
| 实体 | mentions、occurrences、canonicals、aliases、assignments、candidates、cannot-links、review calls |
| 关系 | runs、pair opportunities、source/model calls、proposals、hypotheses、validations、verification attempts |
| 事实 | normalized facts、fact evidence、conflicts、collapse audit、relation endpoints |
| 图控制面 | <code>project_strategy_graphs</code>、generation、policy hash、graph lease |
| 社区 | community run、member、edge 和 narrative 产物 |

### 20.3 历史命名兼容

当前代码中可见：

- <code>entity_shadow_namespaces</code>
- <code>relation_shadow_runs</code>
- 其他 <code>shadow_relation_*</code> / <code>shadow_context_*</code>

它们保存的是现行 Knowledge Forge 处理链的部分不可变数据。保留旧表名的原因是：

- 不改变已经持久化的主键和 evidence reference；
- 避免危险的大规模数据迁移；
- 允许旧运行继续审计和复现。

产品文案、API 策略键和新模块 facade 已使用 Knowledge Forge / grounded 命名。面试时应把它解释成 “存储兼容层”，而不是同时存在两套线上算法。

### 20.4 多项目隔离

所有派生记录带 <code>project_id</code>；图记录还带 <code>project_key</code>。

查询和 mutation 必须同时受以下边界约束：

~~~text
project_id
AND strategy namespace
AND generation or snapshot
~~~

不能只靠全局 entity name 查找，否则相同名称会跨项目串联。

### 20.5 来源链

一条已发布关系可反查：

~~~text
fact
 -> fact_evidence
 -> verification_attempt
 -> hypothesis / proposal
 -> pair opportunity
 -> subject/object occurrence
 -> grounded source map
 -> chunk
 -> content revision
 -> document
 -> immutable source file hash
~~~

因此界面中的 “查看原文” 不是从实体描述中模糊搜索，而是读取已保存的精确来源链。

---

## 21. 一致性、幂等、失败恢复与删除语义

### 21.1 幂等

幂等来自稳定身份和指纹：

- 文件由 SHA-256 定位；
- revision 由内容和处理配置定位；
- chunk 来自稳定来源 key 和处理版本；
- mention / occurrence 包含精确 span；
- entity namespace 包含完整 treatment；
- pair opportunity 包含端点和修订；
- fact key 包含 canonical endpoints 和规范谓词；
- graph 由 generation + policy hash 定位。

重复请求不会靠 “先删除所有数据再赌一次” 实现，而是：

1. 复用已完成且完全匹配的阶段；
2. 只 claim 未完成工作项；
3. 将新结果写入未激活快照；
4. 最后原子切换。

### 21.2 代码升级新鲜度

实体和关系有独立 code treatment：

- 实体代码变化会使 entity namespace 过期。
- 仅关系 contract 变化时，可复用完全一致的 entity namespace。
- 关系 run 的 semantic contract、prompt、Schema、阈值和模型输入都进入 fingerprint。
- 与当前 treatment 不兼容的旧运行不得继续被广告为 ready。

如果旧模型调用输入与当前调用完全一致，可做 code-only replay；否则必须真实重跑关系阶段。

### 21.3 Re-ingestion

重新处理一个文档时：

- 新建内容修订；
- 只替换该文档的 section、chunk、embedding 和来源贡献；
- 项目实体、关系和社区在新快照中重新计算；
- 其他文档对同一事实的证据继续保留；
- 激活前旧快照保持完整，但当前公开图查询在 stale/rebuilding 期间 fail closed，不把旧图冒充成最新图。

### 21.4 删除

删除文档不会直接在 HugeGraph 中按文件暴力删所有相邻边。正确语义是：

1. 文档软删除。
2. 图状态置 stale。
3. 新构建排除该文档。
4. 关系若还有其他文档 evidence，继续发布。
5. 只有证据集合为空的事实才从新 generation 消失。

### 21.5 失败分类

| 类型 | 示例 | 处理 |
|---|---|---|
| 用户输入错误 | 不支持格式、空文件、重复文件 | 请求级明确报错 |
| 解析错误 | 无文本 PDF 且 Docling 不可用 | 文档预处理失败，旧 revision 保留 |
| 模型协议错误 | 非 JSON、截断、缺字段 | 有界重试；不伪装为空结果 |
| 语义拒绝 | 只有共现、方向不成立 | REJECTED 审计 |
| 不确定 | 类型、角色或谓词不足 | UNCERTAIN，不发布 |
| 预算延迟 | pair 超过上限 | DEFERRED_BUDGET |
| 图同步错误 | HugeGraph 不可用 | 新 generation 不激活 |
| CAS 失败 | worker 已过期或被新任务取代 | 丢弃本次激活 |
| 用户终止 | cancel flag 被 worker 检查到 | 安全撤销暂存结果 |

### 21.6 不允许静默回退

当 Knowledge Forge 构建失败时，系统明确失败。不能：

- 读取旧 Stable/Enhanced 路径的结果冒充新图；
- 将旧 entity namespace 与新 relation run 拼接；
- 在原生 GraphRAG 和 KF 之间自动切换；
- 将 verifier 技术失败改成无关系；
- 将 stale 图标为 ready。

---

## 22. 复杂度、成本与长文档优化

### 22.1 主要复杂度

设：

- <code>B</code>：源文件字节数；
- <code>C</code>：graph-eligible chunk 数；
- <code>M</code>：mention 数；
- <code>k</code>：每个 mention 的有界归一候选数；
- <code>n_i</code>：第 i 个原子范围的 occurrence 数；
- <code>E</code>：已验证关系数；
- <code>V</code>：canonical entity 数。

#### 解析

~~~text
O(B)
~~~

表格和 PDF 布局模型会增加常数，但不改变按内容规模线性扫描的基本形式。

#### 分块与来源映射

~~~text
O(B + C)
~~~

#### Embedding

~~~text
O(number_of_retrieval_chunks)
~~~

可批处理，查询通过 HNSW 避免对所有向量做全扫描。

#### 实体模型调用

基础近似：

~~~text
O(C + recursive_splits + gleaning)
~~~

每个 chunk 有字符、句数和返回项上限。

#### 实体归一

不做全量 <code>M²</code> 比较，而是先 lexical/ANN 召回：

~~~text
O(M × k)
~~~

其中 <code>k <= 30</code>，进入昂贵 LLM 复核的候选最多 5。

#### 关系机会

理论局部机会：

~~~text
sum_i n_i(n_i - 1) / 2
~~~

但系统不会执行全项目 <code>V²</code> 模型分类：

- 按 atomic context 限制；
- 句距、字符距离、类型和结构预过滤；
- 每 chunk pair 上限 256；
- source-first 清单覆盖后跳过无关共现 pair；
- verifier 批量 8 条；
- focused review 最多 24。

#### 图同步

~~~text
O(V + E)
~~~

#### 社区检测

Leiden 的实际成本取决于图结构和实现，输入已先合并重复无序实体对并设置 300 秒 timeout。

### 22.2 为什么历史长文档会运行数十小时

早期成本容易被以下步骤放大：

- 目录、父级摘要和参考文献也进入图抽取；
- 每个 pair 都单独做关系分类；
- 对所有 Schema binding 做通用第二轮；
- verifier 响应过长导致批量失败后拆成单条重试；
- parser 将微型标题拆成大量小 chunk；
- 内容、实体和关系的代码变化无法分层复用；
- 服务中断后依赖进程内状态重跑。

当前针对这些问题的通用改造：

- graph eligibility 分流；
- 微型 section coalescing；
- source-first fact inventory；
- binding gleaning 默认关闭；
- 有界 reason code；
- verifier batching；
- 局部 pair ledger 和预算；
- 不可变 work item 续跑；
- 内容、实体和关系分层 fingerprint；
- 关系代码变化时复用实体 namespace；
- Schema 发现从普通构建解耦。

### 22.3 离线工作量预估

在 7 份小于 1 MiB 的真实中文 PDF 上，当前预检得到：

- 62,385 个 graph token；
- 935 个计划基础模型调用；
- 其中实体 374；
- open-fact / Schema binding 561；
- 最大单文档约 230 次基础调用。

该数字不包含：

- 数据相关的显式 Schema induction；
- reference adjudication；
- batched final verification；
- focused review；
- retry。

因此它是工作量下界/基础计划，不是固定耗时 SLA。

### 22.4 可解释的进度

进度不应只显示一个长时间不动的百分比。阶段应拆成：

- 结构解析；
- 分块与向量；
- grounded source manifest；
- 实体 work item 完成数；
- canonical resolution；
- source fact context 完成数；
- pair ledger 状态；
- Schema binding；
- verifier batch；
- fact normalization；
- PostgreSQL 持久化；
- HugeGraph generation；
- 社区。

模型调用数、重试数、terminal failure 和 deferred budget 应作为诊断指标，而不是只在日志中输出 “第 156/156 条”。

---

## 23. 质量评估方法与当前工程基线

### 23.1 为什么不能只看边数

边多可能意味着召回高，也可能意味着：

- 共现泄漏；
- 重复关系；
- 方向错误；
- <code>related_to</code> 滥用；
- 端点不是完整实体；
- 无来源事实。

因此质量评估分成：

1. mention；
2. canonical entity；
3. candidate pair；
4. directed fact；
5. evidence；
6. hard-negative safety；
7. 成本和稳定性；
8. downstream RAG。

### 23.2 标准精确率、召回率与 F1

~~~text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
~~~

关系主指标要求同时匹配：

- directed subject；
- directed object；
- 来源证据；

谓词字符串在独立诱导 Schema 下作为诊断项；但所有发布谓词仍必须属于当前活动 Schema，且不能发布泛化的 <code>related_to</code>。

### 23.3 Mention 指标

- exact span：起止位置完全一致。
- overlap span：预测与标注区间有重叠，采用一对一最大匹配。
- type accuracy：在匹配 mention 上比较类型。

边界选择存在主观性，所以 exact-span F1 是严格诊断；overlap mention precision/recall 更适合判断端点是否实际找回。

### 23.4 Canonical B³

对每个 mention <code>i</code>：

- <code>C(i)</code>：系统预测的 canonical cluster；
- <code>G(i)</code>：Gold cluster。

~~~text
P_B3 =
    (1 / N)
    × sum_i |C(i) intersection G(i)| / |C(i)|
~~~

~~~text
R_B3 =
    (1 / N)
    × sum_i |C(i) intersection G(i)| / |G(i)|
~~~

~~~text
F1_B3 =
    2 × P_B3 × R_B3 / (P_B3 + R_B3)
~~~

B³ 同时惩罚：

- 过度合并导致 cluster 太大；
- 归一不足导致同一实体被拆散。

### 23.5 方向和证据

~~~text
direction_accuracy =
    correctly_oriented_supported_relations
    / supported_non_symmetric_relations
~~~

~~~text
evidence_validity =
    published_relations_with_exact_valid_spans
    / published_relations
~~~

硬负例泄漏率：

~~~text
hard_negative_leak_rate =
    published_declared_hard_negatives
    / declared_hard_negatives
~~~

目标是 0。

### 23.6 标定数据集

截至 2026-08-11 的冻结工程标定集：

| 指标 | 英文 | 中文 | 合计 |
|---|---:|---:|---:|
| 文档 | 6 | 4 | 10 |
| entity mentions | 144 | 113 | 257 |
| canonical entities | 90 | 71 | 161 |
| positive assertions | 80 | 64 | 144 |
| core-direct positives | 71 | 62 | 133 |
| predicate hard negatives | 16 | 16 | 32 |

覆盖：

- 9 个领域；
- 25 种结构标签；
- prose、表格、层级列表、编号条款、要求、coreference 等；
- 24 个下游问题，其中英文 12、中文 12。

### 23.7 数据集的诚实边界

这些标签由单一 assistant expert 完成，不是三人独立人工 Gold：

- 每篇标定文档仅 421–708 UTF-8 bytes；
- 73 个关系类型中 46 个只出现一次；
- 中文 4 篇是短且合成的 fixture；
- 真实 PDF 主要用于工作量和结构回归，并非完整关系 Gold；
- 未标注关系不能一概当成负例，因为开放域文本可能有额外真实事实。

因此：

- 可以用于工程回归和 stop rule；
- 不能声称达到公开 benchmark 的生产级 SOTA；
- 不应为了凑到 200 条而继续由同一个标注者扩写，制造虚假确定性；
- 真正出现新领域失败时，再增加一小组独立复核的文档族。

### 23.8 当前端到端运行结果

隔离项目通过公开构建入口、真实 <code>qwen3.6:latest</code> 和持久任务完成：

- 169 个实体；
- 157 条发布关系；
- 任务约 103 秒完成。

| 指标 | 英文 | 中文 | 合计 |
|---|---:|---:|---:|
| mention surface recall | 92.4% | 91.2% | 91.8% |
| canonical surface coverage | 91.1% | 91.5% | 91.3% |
| core directed endpoint recall | 69.6% | 90.2% | 79.2% |
| grounded endpoint + evidence recall | 69.6% | 90.2% | 79.2% |
| declared hard-negative leaks | 0 | 0 | 0 |

正确理解：

- 79.2% 是当前冻结 core-direct 集上的端点和证据召回；
- 不是所有开放域关系的绝对召回；
- 精度不能把所有未标注边都算错，因此以明确 hard negative、安全门槛和人工抽查形成区间判断；
- 最新运行比前一运行低 5.4 个百分点，但处于预先声明的 0.08 稳定带内；
- 差异主要来自本地模型 frame 波动和保守 under-merge。

### 23.9 结构专项

端到端较强 bucket：

- 英文表格 10/10；
- 中文层级/列表 7/7；
- 中文表格 6/6；
- 英文编号条款 6/8；
- requirements 3/3；
- 英文 section hierarchy 2/2。

真实 ACore 类 PDF 回归：

- 508 个实体；
- 288 条发布关系；
- core OS 子项 17/17；
- task communication 成员 3/3；
- health monitor 层级 3/3；
- multicore 子项 4/4；
- 无泛化谓词边；
- 无非法自环；
- 无把嵌套子项错误直连到最顶层 root。

### 23.10 下游 RAG 结果

24 个冻结问题 top-10：

| 模式 | 成功问题 |
|---|---:|
| vector-only | 22/24 |
| hybrid | 22/24 |
| graph-only | 13/24 |

图谱在 graph-preferred 问题中找回：

~~~text
36 / 43 = 83.7%
~~~

的所需事实。

图上下文字符：

~~~text
7638 / 60858 = 12.6%
~~~

即图谱用约 12.6% 的 vector context 字符提供了结构化事实补充。负关系泄漏和 generic relation 发布均为 0。

这说明图谱的价值不是单独替代 vector，而是在相同混合准确率下提供更短、结构化、可解释的关系证据。

### 23.11 长度扩展

10 份双语 fixture 组合成 1× 和 10×：

~~~text
source_byte_growth = 9.93×
graph_chunk_and_base_call_growth = 9.09×
~~~

工程门槛要求 10× relation-bearing chunks 的调用/工作不超过 15×，当前通过。

### 23.12 当前工程 stop rule

满足以下条件即可结束无止境 prompt 调优：

- 双语 endpoint + evidence >= 0.70；
- 明确 hard safety 全部通过；
- 10× 长度工作量不超过 15×；
- 下游 hybrid 不比 vector-only 总体差超过 2 个百分点；
- graph-preferred top-10 事实召回 >= 70%；
- graph context <= vector context 的 35%；
- 不增加 unsupported answer 或 wrong citation。

截至 2026-08-11，当前闭环已满足工程 stop rule。

以下情况才应重新打开抽取主链：

- 可复现的 hard-safety 错误；
- 成本出现超线性回归；
- 一个有足够样本的语言/结构 bucket 反复低于 0.70；
- 下游 RAG 明确因图谱漏抽或错抽而下降；
- 要对外声称生产级质量，需要独立人工复核和真正 held-out corpus。

---

## 24. 当前已知局限与合理改进方向

### 24.1 英文自由叙述仍弱于中文结构文档

当前英文 free prose、跨句 coreference 和少量 singleton code/section label 是较弱 bucket。原因包括：

- 英文自由句法的谓词变化更多；
- 保守实体 under-merge 使关系端点无法对齐；
- 一句窗口限制跨句事实；
- reference adjudication 只接受唯一指代。

这属于有意的安全取舍。直接放宽跨句连接会快速增加共现假边。

改进应以小型独立复核集驱动，而不是全局放宽。

### 24.2 Canonical under-merge

系统更偏向把不确定简称保持分离。影响：

- 图中可能出现近义或简称节点；
- 一条真实边因为端点未归一而不匹配 Gold；
- 社区可能被轻微割裂。

好处是避免错误合并将两个实体的全部证据和关系永久混在一起。混合 RAG 还能通过 alias lexical/vector 找回分开的 mention。

### 24.3 社区叙述是确定性摘要

当前社区 narrative 可解释、无幻觉，但表达能力不如 LLM 自由总结。未来若引入模型摘要，应：

- 只读取已验证事实和证据；
- 将模型摘要与确定性事实列表分离；
- 保存 prompt、模型、输入事实和版本；
- 不能让摘要反向生成事实。

### 24.4 统一 confidence 不是校准概率

若未来需要真正关系概率：

- 需要独立人工标注开发集；
- 按关系类型和语言做校准；
- 使用 verifier logit、证据数、来源多样性和结构契约特征；
- 用 reliability diagram、ECE 或 Brier score 验证；
- 不能直接把 LLM 自报 confidence 当概率。

### 24.5 GraphRAG Native 的优势没有全部并入

当前只复用了 GraphRAG 的成熟社区检测思想。GraphRAG Native 在大规模主题总结和全局检索上有优势，但它的开放抽取不具备本文同等级的精确关系 evidence gate。

如果继续融合，应保持：

- 算法思想可以复用；
- 运行产物和控制面保持隔离；
- 不能混合两条策略的事实主键或偷偷 fallback。

### 24.6 评测集需要按失败驱动扩充

不是继续随意找文档做大，而是当线上暴露新 bucket 时加入：

- 新文档结构；
- 新语言现象；
- 新关系角色；
- 新 hard negative；
- 对应 downstream question。

最好由至少两名独立标注者和一名 adjudicator 完成关键事实复核。

---

## 25. 面试中应重点讲的设计取舍

### 25.1 为什么不用 “一个 Prompt 输出完整图”

回答要点：

- 实体、关系、归一和验证是不同误差类型。
- 单调用无法区分漏抽、格式截断、身份错误和关系否定。
- 端到端大 Prompt 难以增量复用，长文档成本高。
- 分层后每个阶段可独立 benchmark、retry、缓存和审计。

### 25.2 为什么 Schema 仍有必要

Schema 不再负责 “没有这个谓词就拒绝事实”，而负责：

- 统一语义；
- 定义方向和端点角色；
- 限制 domain/range；
- 给出正反例和结构政策；
- 决定自动发布还是仅审核；
- 支持归一、冲突和查询。

开放载体解决 Schema 覆盖不足，所以 “保留 Schema” 与 “高召回” 不矛盾。

### 25.3 为什么关系生成和验证必须独立

同一个模型在同一上下文中自我检查，容易维护原先假设。独立 verifier：

- 不依赖生成器自报 confidence；
- 明确检查正反方向；
- 分开检查 subject/object role；
- 协议字段不完整就拒绝；
- 技术失败与语义拒绝分离。

### 25.4 为什么保守归一

错误合并的损害是乘法式的：两个实体一旦合并，其全部 mention、关系、社区和检索证据都会污染。

漏合并的损害通常是加法式的：图中多一个节点，但文本和 alias 检索仍可能找回。因此当前产品选择 under-merge 优于 over-merge。

### 25.5 为什么 PostgreSQL 和 HugeGraph 都需要

- PostgreSQL 擅长事务、不可变审计、状态机、向量和复杂过滤。
- HugeGraph 擅长邻居遍历和图形浏览。
- 将 HugeGraph 当唯一真值，会难以表达一次关系的多次验证、多个来源和冲突。
- 将所有交互遍历都压在关系表上，又不利于图产品体验。

### 25.6 为什么社区边要聚合

全局图展示的是社区拓扑。如果两个社区间有 20 条实体事实，画 20 条平行线只会造成视觉噪声。因此概览按无序社区对聚合为一条线，weight 表示已验证事实数量；进入社区或关系详情后再展示底层事实。

### 25.7 为什么不追求 100% 召回

因为最终目标是 hybrid RAG：

- 图负责高价值、方向明确、可验证的关系；
- 文本块保留完整内容；
- 为补最后少量召回而放宽证据门槛，会增加错误关系和下游错误答案；
- 当前 stop rule 已基于真实 downstream 检索，而不是孤立追求图指标。

---

## 26. 可用于面试的项目叙述

### 26.1 STAR 结构

#### Situation

需要把多格式、跨领域、含中文列表和表格的长文档构造成可用于 RAG 的知识库和知识图谱。早期流程存在图关系漏抽、泛化关系、方向错误、长任务数十小时和重启后状态不一致等问题。

#### Task

设计一套通用而非文档特例的抽取与发布策略，在不牺牲证据可追溯性的前提下，提高实体和关系召回，并让长任务可恢复、可取消、可评估。

#### Action

可按自己的实际贡献选择以下内容讲：

- 设计内容修订、处理指纹和项目级 snapshot。
- 将 retrieval-only 与 graph-eligible chunk 分流。
- 建立 source-first 开放事实清单和通用结构投影。
- 将 Schema 改为治理层，并增加 source assertion 开放载体。
- 建立保守实体归一、cannot-link 和有界身份复核。
- 引入独立有序蕴含 verifier、事实归一和冲突门槛。
- 用 generation、lease 和 CAS 做图发布。
- 将 GraphRAG Leiden 作为 BYOG 社区算法而不混用抽取。
- 建立双语小型标定集、hard negatives、10× 压测和 hybrid RAG 闭环。
- 优化 WebGL 图布局、聚合视图、hover 邻域和证据详情。

#### Result

工程验收结果：

- mention recall 91.8%；
- canonical surface coverage 91.3%；
- directed endpoint + evidence recall 79.2%；
- 32 个声明 hard negatives 零泄漏；
- 真实 PDF 结构切片 17/17、3/3、3/3、4/4；
- 10× 输入对应 9.09× 基础工作；
- hybrid 与 vector 都通过 22/24；
- 图提供 83.7% graph-preferred facts，文本量仅为 vector context 的 12.6%。

结论应表述为 “达到当前 hybrid-RAG 工程可用门槛”，不要表述为 “开放域关系抽取已达到生产级 100%”。

### 26.2 这个项目最有技术含量的三个点

1. **不可变、来源约束的事实流水线**：不是让 LLM 直接写图，而是将每次候选、验证和发布都绑定到 source span 和 treatment snapshot。
2. **召回与精度解耦**：source-first、结构投影和开放载体负责召回；ordered verifier、Schema role 和冲突消解负责精度。
3. **端到端工程闭环**：把算法指标、长文档成本、任务状态、图发布和下游 hybrid RAG 放在同一套验收里。

### 26.3 最值得展示的架构决策

如果白板时间有限，画这条链：

~~~text
Immutable Source
  -> Structured Blocks
  -> Retrieval Chunks
  -> Grounded Context
  -> Mentions
  -> Canonicals
  -> Source Facts
  -> Independent Verification
  -> Normalized Fact Snapshot
  -> Graph + Hybrid RAG
~~~

在 “Retrieval Chunks” 到 “Grounded Context” 之间画分叉，强调不是所有可检索内容都能产生图事实。

---

## 27. 高频面试追问与回答

### Q1：如何防止大模型幻觉关系？

不是依靠一句 prompt，而是多重硬门槛：

- 端点必须是当前来源中的精确 occurrence；
- evidence 必须是连续原文并覆盖两端；
- 必须在 atomic scope；
- 生成与验证调用分离；
- verifier 同时检查 forward 和 reverse；
- Schema 检查角色、方向、domain/range；
- 否定、共现、自环和冲突不发布；
- 最终图只读取完整新鲜快照。

### Q2：高召回和高精度如何兼顾？

高召回放在 candidate layer：

- entity inventory + gleaning；
- 精确重复恢复；
- 列表、表格、标签等结构端点；
- source-first open facts；
- 开放 source assertion。

高精度放在 publication layer：

- exact span；
- atomic scope；
- independent ordered entailment；
- role/type/direction；
- normalization/conflict；
- fail closed。

### Q3：为什么不是每个实体 pair 都问模型？

全项目 pair 是 <code>O(V²)</code>，大量只是共现。当前按来源原子范围枚举，先做 source-first 清单，再对局部 pair 使用距离、类型、结构和预算过滤，使成本接近有效 chunks 和局部实体密度，而不是全图实体平方。

### Q4：列表关系如何避免全部被拒绝？

“同级相邻” 被拒绝，但 “直接父子” 被保留。解析器保存 marker、indent、style 和 parent key；结构投影生成 child → parent 候选；发布前从不可变 metadata 重验。这样截图中的功能项都能属于系统，同时不会把 “任务管理” 与 “时间管理” 互相连成关系。

### Q5：为什么要有 <code>source_assertion</code>？

通用 Schema 永远无法预先列出所有领域谓词。开放载体保留原文实际谓词和方向，仍走相同 verifier。它比丢失事实召回更好，也比统一发布 <code>related_to</code> 更有语义。

### Q6：UNCERTAIN 有什么用？

它保存有价值但不满足自动发布条件的候选和原因，可用于：

- 原因分流；
- 人工审核；
- 新 Schema 候选；
- prompt/规则回归；
- 统计哪个 bucket 经常失败。

它不会污染正式图。

### Q7：服务重启会丢构建吗？

用户可见状态在 PostgreSQL，内容和关系 work item 有稳定身份，完成产物在不可变 snapshot。服务重启可以识别未完成 lease 并恢复或失败闭环；图只在 CAS 激活后切换。需要特别避免在运行中的独立 GraphRAG Native child 存在时直接重启，因为其恢复语义不同。

### Q8：图数据库为什么不是唯一数据源？

图数据库适合遍历，不适合完整表达多轮模型调用、多个 evidence span、冲突和快照 CAS。PostgreSQL 是控制面与真值审计，HugeGraph 是物化读模型。

### Q9：如何处理新增文档？

新文档先生成活动内容修订，然后项目图置 stale。构建时所有活动文档形成新项目 snapshot，实体可重新归一，事实按多文档 evidence 归并，社区重新检测；最后一次性激活新 generation。

### Q10：如何评估一个未穷尽标注的开放域图？

不能把所有未标注边当 FP。采用：

- evidence-complete 的 core-direct positives；
- 显式 hard negatives；
- exact endpoint/evidence；
- 抽样人工审查额外开放事实；
- downstream questions；
- 成本和稳定性 gate；
- 对 precision 给出区间而非虚假单点。

### Q11：为什么中文结果比英文更好？

当前中文集更多是层级列表、表格和明确结构，parser contract 能稳定恢复；英文自由叙述、跨句指代和别名变化更复杂。这个差异反映当前数据结构分布，不代表模型普遍更擅长中文。

### Q12：系统最大瓶颈是什么？

主要是本地 LLM 调用，而不是图数据库写入。优化优先级是：

1. 减少无事实价值的 chunk；
2. source-first 代替 pair 全扫描；
3. 批量 verifier；
4. 分层缓存和 replay；
5. 只对高风险项 focused review；
6. 最后才考虑数据库写入微优化。

### Q13：如何确保通用修复而不是针对某文档硬编码？

生产分支只允许基于：

- parser role；
- 通用结构；
- Schema policy；
- 类型和来源证据；
- 可配置阈值；

做决策。具体文档名、社区 ID、领域术语和谓词只能出现在测试、Gold fixture 或说明中。

### Q14：社区描述会不会幻觉？

当前由已验证实体、关系和证据确定性生成，不调用自由总结模型。未来若引入 LLM，也只能作为 presentation adapter，不能反向写事实。

### Q15：怎样证明图谱对 RAG 有价值？

不能只说图更直观。当前下游实验显示：

- hybrid 与 vector 同为 22/24；
- graph-only 只有 13/24，证明图不能替代文本；
- 图在适合关系的问题中提供 83.7% 所需事实；
- 图上下文只有 vector 字符数的 12.6%。

它的价值是结构化、压缩和可解释的关系证据。

---

## 28. 关键默认配置速查

### 28.1 项目与文件

| 配置 | 默认 |
|---|---|
| LLM | <code>qwen3.6:latest</code> |
| Embedding | <code>bge-m3:latest</code> |
| 上传上限 | 200 MB |
| 默认模型上下文预算 | 8192 |
| 默认 chunk 最大 token | 512（项目默认；策略内部另有 max） |

### 28.2 Chunk

| 配置 | 默认 |
|---|---:|
| compatibility max chars | 2400 |
| compatibility overlap chars | 160 |
| general target/max | 512 / 800 |
| general overlap | 12% |
| general coalesce min sections | 24 |
| paper target/max | 640 / 900 |
| paper overlap | 10% |

### 28.3 模型安全

| 配置 | 默认 |
|---|---:|
| Ollama seed | 17 |
| fallback context | 8192 |
| max output | 2048 |
| safety margin | 256 |
| absolute token variance | 128 |
| relative token variance | 50% |
| image token estimate | 1024 |

### 28.4 实体

| 配置 | 默认 |
|---|---:|
| segment chars | 2400 |
| segment sentences | 8 |
| segment items | 96 |
| gleaning | 1 |
| lexical candidates | 10 |
| vector candidates | 20 |
| reranker cap | 30 |
| LLM review candidates | 5 |
| LLM review threshold | 0.68 |

### 28.5 Schema

| 配置 | 默认 |
|---|---:|
| discovery during build | false |
| sample chunks base/ceiling | 12 / 48 |
| sample chars base/ceiling | 12000 / 60000 |
| chars per sample chunk | 2400 |
| per-section sample | 2 |
| corpus batch | 4 |
| candidate confidence | 0.85 |
| single exact evidence confidence | 0.96 |
| novelty threshold | 0.22 |
| max entity/relation additions | 5 / 8 |

### 28.6 关系

| 配置 | 默认 |
|---|---:|
| open fact gleaning | 1 |
| binding gleaning | 0 |
| shortlist per frame | 1 |
| shortlist threshold | 0.18 |
| general lexical frame threshold | 0.34 |
| context open carrier cap | 96 |
| pair sentence window | 1 |
| pair char distance | 384 |
| pair budget/chunk | 256 |
| verifier batch | 8 |
| verifier max chars | 48000 |
| focused review | 24 |
| protocol attempts | 3 |
| secondary recovery | false |

### 28.7 图与社区

| 配置 | 默认 |
|---|---:|
| graph lease | 900 秒 |
| Leiden seed | 42 |
| max cluster size | 10 |
| community timeout | 300 秒 |
| use LCC only | false |

---

## 29. 核心公式速查

### 内容与身份

~~~text
document_hash = SHA256(file_bytes)
config_hash   = SHA256(canonical_json(config))
~~~

### Chunk token

~~~text
T_hat =
    max(
        1,
        N_CJK
        + ceil(1.3 × N_ASCII_word)
        + ceil(N_other_punct / 4)
    )
~~~

### Dense + lexical

~~~text
D = max(0, cosine(q, x))
L = sum_t min(cq(t), cx(t)) / sum_t cq(t)
S = 0.72D + 0.28L
~~~

### Schema sampling

~~~text
doc_MMR =
    0.55 diversity
  + 0.30 information
  + 0.15 new_section
~~~

~~~text
corpus_MMR =
    0.42 diversity
  + 0.28 information
  + 0.22 new_document
  + 0.08 new_section
~~~

### Entity resolution

~~~text
C0 =
    0.52 lexical
  + 0.38 vector
  + 0.10 type
~~~

~~~text
C1 =
    0.34 lexical
  + 0.30 vector
  + 0.26 reranker
  + 0.10 type
~~~

### Pair opportunities

~~~text
P = n(n - 1) / 2
~~~

### Schema relation shortlist

~~~text
R =
    0.36 relation_similarity
  + 0.25 evidence_example_similarity
  + 0.22 endpoint_example_similarity
  + 0.12 role_policy_similarity
  + 0.05 type_score
~~~

### Hybrid graph ranking

~~~text
G =
    0.22 network
  + 0.12 seed_relevance
  + 0.36 relation_semantic
  + 0.15 relation_surface_lexical
  + 0.08 text_score
  + 0.02 lexical
  + 0.05 shared_seed_score
~~~

### UI node size

~~~text
z = log(1 + importance)
u = clip((z - p10) / (p90 - p10), 0, 1)
size = min + (max - min) × u^0.76
~~~

### 质量指标

~~~text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
~~~

---

## 30. 代码导航

| 模块 | 当前职责 |
|---|---|
| <code>knowledge_forge/web.py</code> | HTTP 路由和页面，不放领域算法 |
| <code>knowledge_forge/service.py</code> | 上传、预处理、构建、任务和发布编排 |
| <code>knowledge_forge/database.py</code> | PostgreSQL schema、migration 和持久化 |
| <code>knowledge_forge/chunking.py</code> | 确定性解析后结构与分块 |
| <code>knowledge_forge/media.py</code> | 图片提取、OCR 和多模态描述 |
| <code>knowledge_forge/ollama.py</code> | Ollama transport、预算和 JSON 解析 |
| <code>knowledge_forge/vector_store.py</code> | 向量持久化和 hybrid 基础分数 |
| <code>knowledge_forge/schema_evolution.py</code> | 基础逻辑 Schema、采样、增量治理 |
| <code>knowledge_forge/grounded_context.py</code> | 当前来源上下文 facade |
| <code>knowledge_forge/shadow_context.py</code> | 历史命名下的精确 context/source-map 实现 |
| <code>knowledge_forge/grounded_entities.py</code> | 当前实体处理 facade 和来源约束语义 |
| <code>knowledge_forge/entity_shadow.py</code> | 历史命名下的不可变实体工作流 |
| <code>knowledge_forge/entity_resolution.py</code> | 实体候选、归一和保守 identity resolution |
| <code>knowledge_forge/entity_reranker.py</code> | 实体 pair reranking |
| <code>knowledge_forge/grounded_relations.py</code> | 当前关系 facade |
| <code>knowledge_forge/relation_shadow_runtime.py</code> | 关系运行编排和 semantic contract |
| <code>knowledge_forge/relation_shadow_v2.py</code> | pair、proposal、hypothesis 和归一主链 |
| <code>knowledge_forge/relation_verification.py</code> | 独立有序蕴含验证 |
| <code>knowledge_forge/relation_secondary.py</code> | 原因分流二次复核，默认关闭 |
| <code>knowledge_forge/knowledge_forge_descriptions.py</code> | 实体/关系展示描述 |
| <code>knowledge_forge/hugegraph_store.py</code> | 固定 KF 物理 Schema 和图同步 |
| <code>knowledge_forge/knowledge_forge_communities.py</code> | 当前图适配与社区产物 |
| <code>knowledge_forge/graphrag_community.py</code> | Leiden/GraphRAG BYOG 社区检测 |
| <code>knowledge_forge/graph_retrieval.py</code> | 图关系检索和混合排名 |
| <code>knowledge_forge/graph_explorer.py</code> | 有界图场景、社区和邻域数据 |
| <code>knowledge_forge/graph_manifest.py</code> | 图输入清单和新鲜度 |
| <code>knowledge_forge/graph_policy.py</code> | 图策略版本和哈希 |
| <code>knowledge_forge/graph_safety.py</code> | 图发布安全校验 |
| <code>knowledge_forge/runtime_safety.py</code> | 运行时安全检查 |
| <code>docs/knowledge-forge-evaluation-acceptance-2026-08-11.md</code> | 当前工程验收和 stop rule 依据 |

---

## 31. 最终总结

Knowledge Forge 当前策略的本质不是 “让大模型生成一个图”，而是构建一条可验证的知识生产线：

1. **原始内容不可变**：SHA-256 和内容修订保证来源确定。
2. **结构优先**：章节、列表、表格、页码和媒体在分块前被恢复。
3. **检索与图谱分流**：所有有价值内容可检索，只有可验证内容能生成事实。
4. **实体先落点再归一**：mention 精确锚定，canonical 合并保守。
5. **关系先开放召回再语义治理**：source-first 防漏，Schema 和 verifier 防错。
6. **事实与证据分离**：一个事实可有多份来源，候选和不确定项不污染正式图。
7. **快照式发布**：generation、lease、CAS 和安全取消保证用户不看见半成品。
8. **图和文本共同服务 RAG**：图提供结构化关系，文本保留完整上下文。
9. **以工程闭环收尾**：抽取指标、hard negative、长文档成本和下游检索共同决定是否可用。

面试时最值得强调的一句话是：

> 我们没有把 LLM 当作知识图谱数据库的直接写入器，而是把它当作受来源约束的候选生成器和独立审核器；最终事实由精确证据、Schema 政策、方向验证、冲突消解和不可变快照共同决定。
