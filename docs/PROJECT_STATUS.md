# Knowledge Forge 当前项目状态

> 更新时间：2026-08-21
> 本文只描述当前实现。历史实验结论保存在 `docs/archive/`，不再作为运行说明。

## 产品定位

Knowledge Forge 是通用、多项目的知识库与知识图谱构建平台。系统从用户上传文档开始，完成结构解析、知识分块、向量索引、实体与关系抽取、实体归一、图谱发布、社区发现、社区报告和可追溯检索。

核心运行边界：

- PostgreSQL + pgvector：权威业务数据、来源、任务和向量索引；
- HugeGraph `knowledge_forge`：Knowledge Forge 项目图的可遍历副本；
- Ollama：模型发现、Embedding、结构化抽取、Schema 评估和社区报告；
- 独立 reranker HTTP 服务：有界候选重排；
- `KnowledgeFiles/`：源文件、解析结果、媒体和 GraphRAG Native bundle。

## 当前两种构建策略

### Knowledge Forge

Knowledge Forge 复用已准备的结构块、知识块和向量，构建精确来源约束的规范实体与已验证关系。PostgreSQL 快照完成一致性校验后同步 HugeGraph，再生成社区和异步详细报告。Chunk、实体、关系和已完成报告共同服务检索；报告未完成不会阻塞基础检索。

### GraphRAG Native

GraphRAG Native 独立运行 GraphRAG 3.1.1，实体、关系、社区、报告和查询索引保存在不可变 Parquet/LanceDB bundle。它不写 Knowledge Forge 实体表、pgvector 或 HugeGraph，也不回退到 Knowledge Forge 抽取流程。只有 bundle 验证和来源新鲜度检查均通过的运行才能激活查询。

## 已完成能力

| 领域 | 当前状态 |
| --- | --- |
| 项目、文档、模型与知识语言配置 | 已实现 |
| PDF、DOCX、Markdown、TXT、XLSX、XLS、CSV 解析 | 已实现 |
| 结构块、知识块、页码、图片与表格预览 | 已实现 |
| 增量预处理、向量复用与 pgvector HNSW | 已实现 |
| additive-only 逻辑 Schema 发现与复核 | 已实现 |
| 精确 occurrence 实体抽取和保守实体归一 | 已实现 |
| 列表、表格、正文与开放关系的关系召回 | 已实现 |
| 独立关系验证、方向/角色审计和发布门禁 | 已实现 |
| HugeGraph 项目隔离、代次同步和一致性校验 | 已实现 |
| 社区发现、基础描述、异步详细社区报告 | 已实现 |
| GraphRAG Native Basic/Local/Global/DRIFT 查询 | 已实现 |
| Chunk、实体、关系与社区报告混合检索 | 已实现 |
| PostgreSQL 持久化任务、取消和重启恢复 | 已实现 |
| 图谱、关系审计、来源与媒体交互界面 | 已实现 |
| 跨领域中英文回归基准与完整测试集 | 已建立 |

## 当前质量判断

当前知识图谱已经具备作为 RAG 检索来源之一所需的工程基础：所有可发布关系具有项目、文档、Chunk 和证据来源；实体归一保守；不完整运行不能覆盖旧图；社区报告可回到关系及原文。

质量目标是“可检索、可追溯、可与直接知识库互补”，而不是追求图谱单独覆盖全部文档事实。仍需持续监控的主要问题是：

- 超长、版面破碎或 OCR 较差文档的实体边界；
- 跨段、隐式和高度领域化关系的召回；
- 缩写、译名和同名异义实体的保守拆分；
- 社区报告对稀疏社区的覆盖和生成成本；
- 不同模型、格式和知识语言上的漂移。

这些问题应通过跨领域评估集、构建漏斗和真实问答效果判断，不再以单个文档或关系数量作为唯一指标。

## 代码与兼容状态

当前应用代码采用 `grounded entity`、`grounded relation`、`evidence context` 和 `relation pipeline` 命名。早期 PostgreSQL 表、迁移、索引、运行 ID、Schema 元数据和策略版本中的 `shadow` 是已经持久化的兼容标识，保留它们是为了现有项目可继续读取和恢复，不代表当前存在第三套实验策略。

历史 Codex 任务提示、一次性 pilot 脚本、未被现行入口使用的 BYOG 报告原型和提交到 Git 的大体积评测输出已从源代码仓库移除。评测输出改为本地 `artifacts/` 生成，通用测试使用小型临时夹具。

## 下一阶段

知识构建主链进入稳定维护阶段。下一项产品建设是统一知识问答智能体：一套会话和证据协议，分别接入 Knowledge Forge 与 GraphRAG Native 检索适配器。第一版采用固定单知识库、有界检索、统一 rerank、答案引用和拒答；多轮动态路由与 LangGraph 持久化在证明价值后分阶段加入。

完整路线见 [`knowledge-agent-implementation-roadmap.md`](knowledge-agent-implementation-roadmap.md)。
