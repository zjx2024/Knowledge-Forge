# Knowledge Forge 知识库智能体完整实现方案与阶段路线

> 状态：目标架构与实施基线
> 更新时间：2026-08-21
> 适用范围：Knowledge Forge 与 GraphRAG Native 两种知识构建策略
> 核心原则：一套智能体、统一证据协议、多个检索适配器；回答必须可追溯，知识不足时明确拒答或说明限制。

## 1. 文档目标

本文定义 Knowledge Forge 从“知识库与知识图谱构建平台”扩展为“可检索、可对话、可编排的知识智能体平台”的完整方案，并给出从第一版到最终形态的分阶段交付顺序。

本文同时回答以下问题：

1. 两种构建策略是否需要两套智能体；
2. 智能体何时需要 LangGraph，何时普通 Python 流程更合适；
3. 如何自动或显式选择知识库；
4. 如何统一知识块、实体、关系、社区报告和 GraphRAG 查询结果；
5. 如何进行多路召回、融合、重排、证据充分性判断和答案生成；
6. 如何保证来源引用、项目隔离、权限、安全、成本和可恢复性；
7. 每一阶段的功能边界、验收指标和停止条件。

本文不是立即引入全部框架的“大爆炸”方案。第一版优先复用当前已经稳定的检索能力，后续只有在动态分支、循环检索、人工介入和持久恢复确有价值时引入 LangGraph。

## 2. 当前基础与边界

### 2.1 已有可复用能力

Knowledge Forge 当前已经具备：

- 项目、文档、结构块、知识块和版本化来源；
- PostgreSQL + pgvector HNSW 的知识块和实体向量检索；
- 关键词匹配、实体语义召回和有界两跳关系扩展；
- 实体、关系、页码、原文、图片和表格来源；
- Knowledge Forge 社区与异步详细社区报告；
- GraphRAG Native 的 Local、Global、DRIFT、Basic 查询；
- 项目级知识语言、模型、Embedding 和上下文预算；
- PostgreSQL 持久化任务、取消、恢复和状态查询；
- 独立 reranker HTTP 服务；
- HugeGraph 可浏览物理图和 PostgreSQL 权威事实层。

这些能力意味着第一版智能体不需要重新实现文档解析、Embedding、实体关系抽取或图数据库，也不需要让大模型直接生成 SQL/Gremlin。

### 2.2 当前仍缺少的智能体能力

- 统一的问答 API、会话和消息模型；
- 两种构建策略之上的统一检索结果协议；
- 知识库选择、问题分类和检索路由；
- 跨通道融合与统一 rerank；
- 证据充分性判定和有限次数检索修正；
- 答案级引用校验、冲突提示和拒答策略；
- 多轮指代消解与受控会话记忆；
- 智能体运行轨迹、Token、延迟和质量评估；
- 权限过滤、提示注入隔离和工具调用策略。

### 2.3 明确不做的事情

- 不构建“Knowledge Forge 智能体”和“GraphRAG 智能体”两套产品；
- 不让模型绕过服务层直接访问 PostgreSQL、HugeGraph、文件系统或 Ollama 管理接口；
- 不把社区报告作为检索可用性的单点依赖；
- 不在证据不足时用模型常识补齐项目事实；
- 不把所有问题都交给最昂贵的 Global/DRIFT 路径；
- 不在第一版加入自主写库、删库、重建图谱或外部系统操作。

## 3. 总体架构

```text
用户 / API / Web 对话界面
        │
        ▼
会话与权限层
  - 用户、项目访问范围、会话、消息、预算
        │
        ▼
问题理解与知识库选择
  - 显式选择优先
  - 自动选择只在授权范围内
  - 识别事实型、关系型、全局总结型、比较型、澄清型问题
        │
        ▼
统一检索编排器
  ├─ Knowledge Forge 混合检索适配器
  │    ├─ Chunk 向量/关键词
  │    ├─ 实体召回
  │    ├─ 一至两跳关系
  │    └─ 社区报告
  ├─ GraphRAG Native 适配器
  │    ├─ Basic
  │    ├─ Local
  │    ├─ Global
  │    └─ DRIFT
  └─ 原文与媒体补全适配器
        │
        ▼
统一证据层
  - 去重、版本校验、融合、rerank、来源多样性、预算裁剪
        │
        ▼
证据充分性与冲突检查
  ├─ 足够：生成答案
  ├─ 可恢复：改写或扩大一次检索
  ├─ 需要用户选择：澄清
  └─ 不足/冲突：带限制说明的拒答
        │
        ▼
答案生成与引用校验
  - 逐项引用、页码/章节/图片/表格、禁止无来源关键事实
        │
        ▼
流式输出、反馈、审计与离线评估
```

## 4. 一套智能体如何支持两种策略

两种策略只在“检索适配器”层不同，智能体的会话、路由、证据融合、答案生成、引用和评估全部共用。

| 能力 | Knowledge Forge | GraphRAG Native | 智能体统一方式 |
| --- | --- | --- | --- |
| 细节文本 | Chunk 混合检索 | Basic / TextUnit | 统一为 `text_evidence` |
| 实体定位 | 规范实体 + 别名 + 向量 | Local 实体上下文 | 统一为 `entity_evidence` |
| 关系查询 | 已验证关系 + 两跳扩展 | Local / DRIFT 关系上下文 | 统一为 `relation_evidence` |
| 宏观总结 | KF 详细社区报告 | Global 社区报告 | 统一为 `community_evidence` |
| 原文来源 | 文档、Chunk、页码、媒体 | Native source map | 统一为 `source_ref` |
| 图版本 | graph generation + run ID | immutable bundle run ID | 统一为 `snapshot_ref` |

智能体不能假设两种策略的实体 ID、社区 ID 或分数可以直接比较。适配器先将各自结果转换为统一证据，再由融合层按排名和证据质量处理。

## 5. 知识库选择

### 5.1 产品模式

支持三种模式：

1. **固定知识库**：用户或业务应用预先指定一个项目。第一版默认采用，行为最可控。
2. **多选知识库**：用户明确勾选若干项目，智能体只在这些项目内检索。
3. **自动选择**：智能体只能在用户已授权项目中选择，且必须返回选择理由和实际检索项目。

任何模式都不能越过权限边界。模型建议的项目 ID 只能作为候选，最终由服务端校验。

### 5.2 自动选择算法

为每个可访问项目建立轻量路由文档：项目名称、说明、领域标签、代表实体、代表 Chunk、社区标题和知识语言。路由文档单独向量化，不把完整项目内容塞给模型。

候选项目分数：

```text
project_score =
    0.45 × project_dense_similarity
  + 0.25 × project_lexical_similarity
  + 0.20 × representative_entity_match
  + 0.10 × recent_user_selection_prior
```

规则：

- 显式选择时不运行自动选择；
- 自动选择最多保留 3 个项目；
- 第一名低于校准阈值，或前两名差值过小时，请用户澄清；
- 跨项目比较问题允许多个项目，普通事实问题优先一个项目；
- 选择结果写入运行轨迹，但不写入长期偏好，除非用户明确保存。

## 6. 问题理解与检索路由

### 6.1 问题类型

路由器输出受约束 JSON，而不是自由文本：

```json
{
  "intent": "fact|entity|relation|global_summary|comparison|procedure|follow_up",
  "scope": "local|global|cross_project",
  "entities": [],
  "time_constraints": [],
  "needs_graph": true,
  "needs_community_reports": false,
  "needs_raw_text": true,
  "ambiguities": [],
  "query_variants": []
}
```

### 6.2 路由优先级

路由采用“确定性规则优先、模型补充”的方式：

- 包含明确实体名、编号、接口、条款或局部关系：KF 实体/关系或 GR Local；
- “整体、主要主题、共同问题、全局趋势、总结全部文档”：社区报告或 GR Global；
- “为什么、如何、步骤、条件”：Chunk + 关系，必要时补社区上下文；
- 比较两个对象：分别检索对象证据，再做交集/差异组织；
- 模糊代词或省略：先用会话上下文改写，不直接扩大到所有知识库；
- 纯文本查找：优先 KF Chunk 或 GR Basic；
- 复杂探索问题：后续版本才允许 DRIFT 或查询分解。

### 6.3 策略路由不是固定映射

GraphRAG 的 Basic、Local、Global、DRIFT 是可复用的检索思想与执行工具，不是智能体必须暴露给普通用户的四个按钮。智能体根据问题和成本选择；高级用户可以显式指定诊断模式。

## 7. 统一检索工具协议

所有工具必须返回 JSON 安全结构，禁止返回数据库连接、任意对象或未经裁剪的模型 Prompt。

```python
class RetrievalRequest(TypedDict):
    run_id: str
    user_id: str
    project_ids: list[int]
    question: str
    query_variants: list[str]
    top_k: int
    token_budget: int
    snapshot_constraints: dict[str, str]

class EvidenceItem(TypedDict):
    evidence_id: str
    evidence_type: str
    project_id: int
    strategy: str
    snapshot_ref: dict[str, str | int]
    title: str
    content: str
    entity_keys: list[str]
    relation_keys: list[str]
    source_refs: list[dict]
    retrieval_channels: list[str]
    raw_scores: dict[str, float]
    fused_score: float
    rerank_score: float | None
    provenance_valid: bool
```

首批工具：

- `search_knowledge_forge(project_id, query, top_k)`；
- `query_graphrag_native(project_id, mode, query)`；
- `load_source_context(project_id, source_ref)`；
- `load_entity_neighborhood(project_id, entity_key, hops, limit)`；
- `load_community_report(project_id, community_key)`。

工具参数由服务端构造和校验。LLM 不能传入 SQL、文件路径、任意 URL 或不在授权列表中的项目 ID。

## 8. Knowledge Forge 检索适配器

### 8.1 当前可直接复用的召回

1. Chunk：pgvector HNSW + 词面匹配；
2. 实体：名称/别名词面召回 + 规范实体向量；
3. 关系：实体种子、一跳关系、邻居二跳关系；
4. 社区：已完成报告的标题、摘要和全文向量；
5. 来源：文档、页码、章节、图片、表格。

当前 Chunk/实体混合分数为：

```text
hybrid_score = w_dense × dense_score + (1 - w_dense) × lexical_score
```

其中余弦相似度为：

```text
cos(q, d) = (q · d) / (||q||₂ × ||d||₂)
```

当前词面分数是查询词项覆盖率：英文/数字使用长度至少为 2 的词项，中文使用二元字符片段。

当前关系排序继续复用项目已有公式：

```text
relation_graph_score =
    0.22 × network_scope_score
  + 0.12 × seed_relevance
  + 0.36 × relation_semantic_score
  + 0.15 × relation_surface_score
  + 0.08 × text_evidence_score
  + 0.02 × lexical_score
  + 0.05 × shared_seed_score
```

智能体第一版不修改上述生产公式，只在适配器外增加统一融合和答案上下文预算。

### 8.2 社区报告降级

- 报告已生成：作为宏观证据参与召回；
- 报告生成中：继续使用 Chunk、实体和关系；
- 报告失败或无资格生成：不阻塞问答；
- 全局总结问题且报告覆盖不足：答案必须说明覆盖范围；
- 报告引用的关系仍需回溯到原文，不能只引用报告文本。

## 9. GraphRAG Native 检索适配器

| 模式 | 适用问题 | 智能体使用策略 |
| --- | --- | --- |
| Basic | 文本事实、基线问答 | 低成本文本通道，可与 KF Chunk 对齐 |
| Local | 特定实体、关系和局部事实 | 实体型/关系型问题优先 |
| Global | 整体主题、全库总结 | 只在全局问题且报告完整时启用 |
| DRIFT | 需要从宏观主题逐步深入的探索问题 | 后续版本启用，设置严格轮次和 Token 上限 |

GraphRAG Native 当前查询本身会调用 LLM 生成答案。接入统一智能体时有两种方式：

1. 第一阶段把其 `context_records` 作为证据，把原答案仅作为候选草稿；
2. 若某模式无法返回足够细粒度上下文，则保留原答案并明确标记为 `derived_answer`，统一答案生成器不得把它伪装成原文。

最终推荐第一种：统一智能体负责最后答案，保证两种策略具有一致的引用、拒答和风格。

## 10. 多路召回融合与重排

### 10.1 归一与去重

不能直接比较不同通道的原始相似度。先按通道保留排名，再以稳定来源键去重：

```text
dedup_key = project_id + snapshot_ref + source_type + source_record_id + span
```

同一原文被 Chunk、关系和社区报告重复召回时，合并 `retrieval_channels`，但保留关系或报告提供的结构语义。

### 10.2 RRF 初步融合

```text
RRF(d) = Σ_channel weight(channel) / (k + rank_channel(d))
```

建议初值 `k=60`，权重通过评估集校准，不声明为行业固定值。初始通道权重：

- Chunk/TextUnit：1.0；
- 实体：0.8；
- 已验证关系：1.1；
- 有完整原文引用的社区报告：0.9；
- 仅派生答案：0.4。

### 10.3 reranker

对 RRF Top-40 使用现有独立 reranker 服务，输入为“问题 + 有界证据文本”。不对完整文档或全部图节点重排。

最终排序建议：

```text
final_score =
    0.50 × rerank_score
  + 0.25 × normalized_rrf
  + 0.15 × provenance_quality
  + 0.10 × source_diversity_gain
```

若 reranker 不可用，明确记录降级并使用 RRF；不将服务故障解释为“没有知识”。

### 10.4 上下文选择

使用边际效用选择而非简单 Top-K：

```text
marginal_utility(d) = final_score(d)
                      - λ_duplicate × max_similarity(d, selected)
                      + λ_coverage × new_question_aspect(d)
                      + λ_source × new_source_bonus(d)
```

约束：

- 证据总 Token 不超过本次预算；
- 每个项目、文档和社区有上限，防止单一来源垄断；
- 关键关系证据必须连同端点名称、谓词、方向和原文；
- 图片和表格只传描述、必要结构和可点击引用，不把二进制放入 Prompt；
- 所有证据必须来自同一已验证快照，禁止跨代混合。

## 11. 证据充分性、冲突与检索循环

### 11.1 充分性评分

充分性不是模型主观置信度。建议由可解释信号组成：

```text
sufficiency =
    0.30 × answer_aspect_coverage
  + 0.25 × top_evidence_relevance
  + 0.20 × provenance_completeness
  + 0.15 × independent_source_support
  + 0.10 × entity_relation_connectivity
  - conflict_penalty
```

每项保留原始值。阈值必须用本项目问答评估集校准。

### 11.2 处理结果

- `SUFFICIENT`：生成答案；
- `RETRYABLE_GAP`：最多进行一次查询改写或补充通道；
- `AMBIGUOUS_SCOPE`：请求用户澄清；
- `CONFLICTING_EVIDENCE`：展示冲突来源，不擅自裁定；
- `INSUFFICIENT`：明确说明当前知识库没有足够依据。

第一版禁止无限循环。最终版默认最多 2 个检索回合、每回合有限工具数，并由总 Token、时间和工具调用预算共同终止。

## 12. 答案生成与引用

### 12.1 生成契约

答案模型只接收：系统规则、用户问题、必要会话摘要、结构化证据和输出 Schema。文档文本始终标记为“不可信数据”，其中的指令不得改变系统行为。

输出结构：

```json
{
  "answer": "...",
  "claims": [
    {
      "text": "...",
      "evidence_ids": ["ev-..."],
      "status": "supported|qualified|conflicting"
    }
  ],
  "limitations": [],
  "follow_up_questions": []
}
```

### 12.2 引用校验

生成后执行确定性校验：

- 每个关键事实至少引用一个存在的 `evidence_id`；
- 引用项目、快照、文档和来源记录必须一致；
- 页码、章节、图片、表格链接由服务端来源对象生成，不由模型拼接；
- 关系方向必须与证据项一致；
- 引用社区报告时同时展示报告引用的原文来源；
- 无来源数字、时间、否定结论和比较结论触发修复或删除；
- 修复失败则返回限制说明，不输出未支持内容。

## 13. 会话记忆

### 13.1 短期记忆

短期记忆只服务当前会话：最近消息、已选择项目、已确认实体、上一轮问题的规范化表达和答案引用。长会话通过摘要压缩，原消息仍保存在 PostgreSQL。

### 13.2 长期记忆

长期记忆只保存用户明确要求保留的偏好或事实，例如回答语言、默认知识库和输出格式。不得自动把模型推断、知识库正文或敏感内容写成用户记忆。

长期记忆与知识库完全分开：

- 知识库事实来自项目文档；
- 会话记忆来自用户交互；
- 两者在 Prompt 中使用不同标签和优先级；
- 删除会话或用户时可独立清理，不影响项目知识。

## 14. LangGraph 的引入边界

### 14.1 第一版不强制使用

第一版是固定有界流程：校验权限 → 检索 → 融合 → 生成 → 引用校验。普通 Python 服务更简单、可测、容易定位问题，不需要为了框架而引入 LangGraph。

### 14.2 从第二阶段开始使用的原因

当系统出现以下需求时引入 LangGraph：

- 根据问题动态选择多个检索分支；
- 证据不足时有限循环改写和补检索；
- 多轮会话状态和流程检查点；
- 人工确认知识库范围、敏感操作或高风险输出；
- 进程中断后从节点边界恢复；
- 查看每个节点的输入、输出、耗时和错误。

LangGraph 官方持久化机制以 `thread_id` 管理检查点，可用于故障恢复、人工介入和会话记忆；中断恢复可能重新执行节点，因此产生副作用的步骤仍必须幂等。Knowledge Forge 生产环境应使用 PostgreSQL 持久化 checkpointer，不使用内存 checkpointer。

### 14.3 建议状态

```python
class AgentState(TypedDict):
    run_id: str
    thread_id: str
    user_id: str
    question: str
    normalized_question: str
    authorized_project_ids: list[int]
    selected_project_ids: list[int]
    intent: dict
    retrieval_plan: list[dict]
    evidence: list[EvidenceItem]
    sufficiency: dict
    retry_count: int
    token_usage: dict
    answer: dict | None
    errors: list[dict]
```

### 14.4 建议节点

```text
START
  → authorize
  → resolve_conversation_context
  → select_projects
  → classify_question
  → plan_retrieval
  → retrieve (并行子任务)
  → fuse_and_rerank
  → judge_sufficiency
      ├─ sufficient → draft_answer
      ├─ retryable and retry_count < limit → rewrite_query → retrieve
      ├─ ambiguous → interrupt_for_clarification
      └─ insufficient/conflict → constrained_answer
  → validate_citations
      ├─ valid → persist_and_stream
      └─ repairable once → repair_answer → validate_citations
  → END
```

## 15. PostgreSQL 数据模型

建议新增，均携带项目/用户边界和时间字段：

### 15.1 会话层

- `agent_threads`：会话、用户、默认项目范围、标题、状态；
- `agent_messages`：角色、内容、结构化附件、Token、创建时间；
- `agent_thread_summaries`：摘要版本、覆盖消息范围、模型和来源哈希；
- `agent_user_memories`：显式长期偏好、命名空间、来源和删除状态。

### 15.2 运行层

- `agent_runs`：问题、状态、选中项目、预算、最终答案和错误；
- `agent_run_steps`：节点、attempt、输入摘要哈希、输出、耗时和状态；
- `agent_tool_calls`：工具、参数摘要、项目、快照、返回数量和错误；
- `agent_evidence`：统一证据、来源、分数、使用状态；
- `agent_claim_citations`：答案 claim 到证据的映射；
- `agent_feedback`：用户评分、问题类型和纠正意见。

### 15.3 LangGraph 检查点

使用独立表或独立 Schema，不能与业务 `jobs` 混为一张表。`jobs` 管理知识构建任务；checkpointer 管理问答图状态。保留 `run_id ↔ thread_id ↔ checkpoint_id` 映射方便审计。

## 16. API 设计

第一版：

```text
POST   /api/agent/threads
GET    /api/agent/threads/{thread_id}
POST   /api/agent/threads/{thread_id}/messages
GET    /api/agent/runs/{run_id}
POST   /api/agent/runs/{run_id}/cancel
GET    /api/agent/runs/{run_id}/events      # SSE
POST   /api/agent/runs/{run_id}/feedback
GET    /api/agent/evidence/{evidence_id}
```

消息请求必须包括显式模式：

```json
{
  "question": "...",
  "project_selection": {
    "mode": "fixed|selected|automatic",
    "project_ids": [1]
  },
  "response_language": "project|zh|en",
  "diagnostic_mode": false
}
```

SSE 事件只发送阶段、增量答案和安全元数据，不泄露隐藏 Prompt、数据库凭据或完整模型原始响应。

## 17. Web 产品设计

### 17.1 第一版

- 项目页增加“知识问答”；
- 默认固定当前项目；
- 展示流式回答、引用卡片、页码、章节和媒体；
- 点击引用打开原文片段，图片和表格可放大；
- 显示“本次使用：知识块/实体关系/社区报告”；
- 提供停止生成、复制、反馈和重新检索。

### 17.2 后续版本

- 多知识库选择器与自动选择解释；
- 可折叠检索轨迹，不展示内部思维链；
- 冲突证据并列视图；
- 会话历史和受控长期偏好；
- 管理员查看延迟、成本、失败分类和评估趋势。

## 18. 安全设计

1. 服务端在每次工具调用前重新校验用户和项目权限；
2. 文档、报告和历史消息全部视为不可信输入，不能覆盖系统规则；
3. 工具采用固定枚举和类型 Schema，禁止任意代码、SQL、Gremlin、URL 和路径；
4. 原文下载和媒体地址使用受控 ID，不暴露本地绝对路径；
5. 日志对 Prompt、正文、密码、Token 和个人数据脱敏；
6. 检索结果必须带 project 和 snapshot，融合时再次验证；
7. 第一版智能体为只读；任何写入、重建或外部操作必须未来单独授权并设置人工确认；
8. 超时、取消和进程恢复不能重复产生外部副作用；
9. 对模型输出执行 JSON Schema、长度、引用和允许字段校验；
10. 可疑提示注入只作为知识内容返回，不作为工具指令执行。

## 19. 性能与成本

### 19.1 预算

每次运行设置：

- 总耗时预算；
- LLM 输入/输出 Token 预算；
- Embedding 与 reranker 调用预算；
- 最大项目数、查询变体数、检索回合数和工具调用数；
- 每个证据类型和每个文档的上限。

### 19.2 并行与缓存

- 同一回合的独立检索通道可以并行；
- 同一查询向量只生成一次；
- 查询规范化、项目路由向量和稳定社区报告可缓存；
- 缓存键必须包含项目、策略、图/Bundle 版本、模型和检索策略版本；
- 缓存不跨用户权限复用未经裁剪的结果；
- Global/DRIFT 设置独立并发限制，避免和构建任务争抢本地 LLM/GPU。

### 19.3 降级顺序

```text
reranker 故障 → 使用 RRF
社区报告未完成 → 使用 Chunk + 实体 + 关系
图谱不可用 → 使用当前可验证的 Chunk 检索并提示降级
单个检索通道超时 → 保留其他通道并记录不完整
答案模型失败 → 返回可读证据列表，不伪造答案
```

## 20. 评估体系

### 20.1 数据集

在现有跨领域文档基准上新增问答集，至少覆盖：

- 中英文；
- PDF、Word、Markdown、表格；
- 单文档、多文档、跨项目；
- 事实、关系、步骤、比较、全局总结、无答案和冲突问题；
- 图片/表格证据；
- 恶意提示注入文本；
- KF 与 GR 均可回答、仅一方可回答和都不可回答的样本。

标注由模型先生成候选、人工复核；没有必要追求唯一标准答案，重点标注可接受事实点、必要来源和不可接受陈述。

### 20.2 离线指标

- Retrieval Recall@K、MRR、nDCG；
- 关键证据覆盖率和来源定位正确率；
- Answer claim precision / recall；
- 引用正确率、引用完整率和页码正确率；
- 无答案识别率、冲突识别率；
- 知识库选择准确率和路由准确率；
- 平均/尾部延迟、Token、模型调用数和失败率；
- KF、GR、融合方案的配对对比。

### 20.3 第一版建议验收线

这些是项目初始目标，必须由真实问答集校准：

- 引用记录可解析率 100%；
- 跨项目数据泄漏 0；
- 关键 claim 引用完整率 ≥ 95%；
- 引用来源内容支持率 ≥ 90%；
- 无答案问题正确拒答率 ≥ 85%；
- P95 首字延迟和完整延迟满足部署硬件基线；
- 相比纯 Chunk RAG，融合方案的关键证据 Recall@10 不下降；
- 任一新路由带来的质量收益应大于其稳定成本，否则不启用。

## 21. 分阶段实施路线

### 阶段 0：接口和评估基线

目标：不改变现有检索结果，先建立可演进边界。

交付：

- 统一 `EvidenceItem`、`SourceRef`、`SnapshotRef`；
- KF 和 GR 只读适配器；
- 固定问答评估集与基线脚本；
- 运行、证据和引用数据表；
- 权限校验和运行预算模型。

验收：同一查询通过适配器前后返回的原始证据集合等价；所有证据都能回到当前快照和来源。

### 第一版：单知识库、固定流程、可引用问答

目标：尽快形成真正可用的知识问答，不引入不必要的自主性。

范围：

- 用户固定选择一个项目；
- KF 默认使用 Chunk + 实体 + 关系 + 已完成社区报告；
- GR 允许显式选择 Basic/Local/Global，普通入口由简单规则选择；
- RRF + 有界 reranker；
- 一次答案生成和一次引用校验；
- 流式回答、来源卡片、停止和反馈；
- 普通 Python 编排，不强制 LangGraph；
- 无多轮长期记忆、无自动跨库、无循环工具调用。

完成标准：通过第一版验收线，并完成真实 HTTP/SSE、PostgreSQL 持久化和 Ollama 模型调用验证。

### 第二版：多轮会话与统一动态路由

目标：处理上下文追问和不同类型问题，同时保持流程有界。

新增：

- 会话、消息和摘要；
- 问题分类、指代消解和检索计划；
- KF/GR 通道动态选择；
- 一次证据不足改写；
- PostgreSQL LangGraph checkpointer；
- 节点级状态、重试、取消和恢复；
- 用户可查看实际使用的知识库与来源。

完成标准：服务重启后可恢复未完成会话运行；追问评估优于无会话基线；工具调用次数受预算约束。

### 第三版：多知识库选择与复杂问题分解

目标：支持多个授权知识库和跨文档/跨项目比较。

新增：

- 显式多选和自动知识库选择；
- 子问题分解、按项目并行检索、跨项目证据隔离；
- 比较型答案模板和冲突证据处理；
- 社区报告覆盖感知；
- GR DRIFT 作为高成本探索工具；
- 选择不确定时 LangGraph interrupt 请求用户确认。

完成标准：自动选择准确率达到校准阈值；跨项目泄漏为零；复杂问题质量显著优于单轮统一查询。

### 第四版：可靠性、个性化与运营闭环

目标：具备长期运行和持续调优能力。

新增：

- 用户明确授权的长期偏好；
- 在线反馈进入待审核评估集；
- 路由、检索、引用和拒答质量看板；
- Prompt/模型/检索策略 A/B 与回滚；
- 成本和延迟自适应路由；
- 高风险领域人工确认接口；
- 故障注入、容量和并发测试。

完成标准：策略变更必须通过离线门禁和灰度；线上失败可定位到具体节点与工具；长期记忆可解释、可删除。

### 最终版：受控的知识工作智能体平台

最终形态不是“无限自主智能体”，而是可配置、可审计的知识工作流平台：

- 统一支持 KF 与 GR，允许新增检索后端而不改变上层协议；
- 根据问题、权限、质量、延迟和成本制定可解释计划；
- 对复杂问题进行有限分解和并行检索；
- 对每个结论提供可核验来源，识别不足和冲突；
- 支持会话恢复、人工介入、审计、评估和回滚；
- 默认只读，写操作作为独立受控工具集逐项授权；
- 不依赖某一家 LLM 或编排框架才能访问核心知识。

## 22. 推荐代码边界

```text
knowledge_forge/
├── agent/
│   ├── contracts.py          # State、Evidence、Source、API Schema
│   ├── authorization.py      # 用户与项目访问范围
│   ├── project_router.py     # 知识库选择
│   ├── query_planner.py      # 问题理解与检索计划
│   ├── retrieval/
│   │   ├── knowledge_forge.py
│   │   ├── graphrag_native.py
│   │   └── source_context.py
│   ├── fusion.py             # 去重、RRF、rerank、预算选择
│   ├── sufficiency.py        # 充分性和冲突
│   ├── answer.py             # 结构化答案生成
│   ├── citations.py          # 确定性引用校验
│   ├── workflow.py           # V1 固定流程
│   └── graph.py              # V2+ LangGraph 编排
├── database.py               # 仅持久化，不放智能体决策
├── service.py                # 应用编排入口
└── web.py                    # HTTP/SSE 与展示
```

智能体模块只能通过现有服务/数据库的受控查询接口读取知识，不应复制一套知识构建逻辑。

## 23. 实施顺序与当前建议

建议下一轮从“阶段 0 + 第一版”开始，先交付单项目、固定流程、证据可引用问答。不要立即引入多智能体、自动跨库和 DRIFT 循环。

第一版稳定后，再根据真实评估结果决定：

- 是否需要 LangGraph；
- 哪些问题确实受益于社区报告或 GR Global；
- 自动知识库选择是否比用户显式选择更有价值；
- 一次改写能否带来足够召回收益；
- reranker 是否应同时服务 Chunk、关系和报告。

每个新增节点都必须证明质量收益或可靠性价值，否则保持简单流程。

## 24. 参考依据

- LangGraph Persistence：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Interrupts：<https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph Memory：<https://docs.langchain.com/oss/python/langgraph/add-memory>
- Microsoft GraphRAG Query Overview：<https://microsoft.github.io/graphrag/query/overview/>
