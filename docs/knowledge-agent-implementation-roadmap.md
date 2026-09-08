# Knowledge Forge 知识库智能体完整实现方案与阶段路线

> 状态：阶段 0、第一版和第二版单知识库范围已实现；第三版多知识库范围尚未开始
> 更新时间：2026-09-01
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

单知识库问答所需的统一 API、会话、证据协议、动态检索、融合重排、充分性判断、一次补检索、引用校验、拒答、多轮语境与运行审计已经实现。后续范围主要是：

- 用户显式多选知识库与授权范围内的自动知识库选择；
- 跨知识库问题分解、并行检索、比较和冲突组织；
- GR DRIFT 的高成本探索分支及更细的并发/成本治理；
- 路由、召回、引用和拒答质量的持续评估与线上观测；
- 经用户明确授权的长期偏好，以及高风险流程的人工确认。

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
  │    ├─ 受控社区联系导航
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

当前路由器输出受约束且可审计的检索计划，而不是自由文本：

```json
{
  "protocol": "knowledge-agent-retrieval-plan-v1",
  "activation": "active",
  "intent": "fact_lookup|entity_explanation|relation_explanation|procedure|comparison|global_summary|source_lookup|visual_lookup|ambiguous",
  "scope": "local|multi_entity|global",
  "family_limits": {"text": 16, "entity": 6, "relation": 12, "community": 6},
  "native_mode": "basic|local|global",
  "retry_allowed": true,
  "referenced_terms": [],
  "reason_codes": ["cue:procedure"]
}
```

计划由通用的中英文问法特征和知识范围确定性生成，不再额外调用 LLM。低置信度问题回到带安全下限的均衡计划，任何一次分类都只调整各证据族预算，不会直接关闭文本、实体、关系或社区证据族。原始问题、独立检索问题、计划和实际执行配额都写入运行审计。

### 6.2 路由优先级

多轮语境模型先把追问补全为独立问题；检索路由本身采用确定性规则：

- 包含明确实体名、编号、接口、条款或局部关系：KF 实体/关系或 GR Local；
- “整体、主要主题、共同问题、全局趋势、总结全部文档”：提高社区证据预算；GR 对真正的多文档/项目级总结使用 Global，单文档总结自动使用 Local；
- “为什么、如何、步骤、条件”：Chunk + 关系，必要时补社区上下文；
- 比较两个对象：分别检索对象证据，再做交集/差异组织；
- 模糊代词或省略：先用会话上下文改写，不直接扩大到所有知识库；
- 纯文本查找：优先 KF Chunk 或 GR Basic；
- 复杂探索问题：后续版本才允许 DRIFT 或查询分解。

对“图 1 / Figure 2 / 表 3 / Table 4”一类带明确编号的视觉问题，KF 还会从当前文档 revision 中做一次有界的精确编号恢复。该通道只恢复已有 Chunk 及其图片/表格来源，不调用 LLM 或 Embedding，也不能绕过活动文档、来源和快照门禁。

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
4. 社区联系：在必要时从事实桥或内容导航关联选择少量相邻社区，再回取底层关系；
5. 社区：已完成报告的标题、摘要和全文向量；
6. 来源：文档、页码、章节、图片、表格。

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

### 8.3 社区联系导航二级召回

社区联系用于扩大候选范围，不新增一种可引用事实。当前实现分为两类路由信号：

1. **跨社区事实桥**：两个社区之间确实存在已验证实体关系。全局图中的聚合线只负责说明关系数量和构成，智能体最终读取的是其底层关系断言。
2. **内容导航关联**：两个没有事实桥的社区在代表实体向量上高度相近。该关联只能帮助选择下一个社区，永远不能作为答案证据。

导航不是每次查询都执行。系统先完成原有 Chunk、实体、两跳关系和社区报告召回，随后在以下任一条件成立时启动有界二级召回：

- 查询映射到多个叶子社区；
- 首轮返回的关系少于 4 条；
- 没有可靠实体种子，但命中了绝对相关度不低于 0.50 的社区报告。

实体种子必须是查询中的明确名称，或实体混合分数不低于 0.48。没有可靠实体或报告种子时不使用社区导航，避免从任意社区开始扩散。

事实桥路由分数为：

```text
S_fact_link = clip₀¹(
    0.50 × S_seed
  + 0.25 × log(1 + fact_count) / log(9)
  + 0.15 × S_relation_lexical
  + 0.10 × I(target_is_also_a_seed)
)
```

内容导航路由分数为：

```text
S_topic_link = min(0.90,
    0.45 × S_seed
  + 0.40 × S_association
  + 0.15 × S_representative_name_lexical
)
```

目标社区中的真实关系再按以下分数选择：

```text
S_routed_relation = clip₀¹(
    0.76 × S_link
  + 0.24 × S_relation_lexical
  + 0.12 × I(touches_representative_entity)
)
```

最后执行四层约束：

1. 默认最多选择 3 个社区联系、8 条关系，可由 `KNOWLEDGE_AGENT_COMMUNITY_NAVIGATION_MAX_COMMUNITIES` 和 `KNOWLEDGE_AGENT_COMMUNITY_NAVIGATION_MAX_RELATIONS` 调整；
2. 每个联系先保留一条关系，再填充剩余预算，避免一个大社区耗尽候选；
3. 所有关系 ID 必须重新通过当前 namespace、当前 relation run、`ELIGIBLE` membership、独立 `SUPPORTED` 验证、活动文档/Chunk 和来源完整性查询；
4. 只有通过上述门禁的底层关系才能以 `relation` 证据进入 RRF 和 reranker；聚合边、内容导航边及其相似度均不创建 `EvidenceItem`。

该流程复用查询已有结果，不新增 LLM 或 Embedding 调用。导航失败会记录为降级并继续使用原有检索，不会导致问答不可用。每次运行保存触发原因、种子社区、实际使用的联系、通过门禁的关系数及 `synthetic_evidence_count=0`，便于评估它是否真正改善召回。

## 9. GraphRAG Native 检索适配器

| 模式 | 适用问题 | 智能体使用策略 |
| --- | --- | --- |
| Basic | 文本事实、基线问答 | 低成本文本通道，可与 KF Chunk 对齐 |
| Local | 特定实体、关系和局部事实 | 实体型/关系型问题优先 |
| Global | 整体主题、全库总结 | 自动模式只用于真正的多文档/项目级总结；用户显式选择时保持 Global |
| DRIFT | 需要从宏观主题逐步深入的探索问题 | 后续版本启用，设置严格轮次和 Token 上限 |

GraphRAG Native 当前查询本身会调用 LLM 生成答案。接入统一智能体时有两种方式：

1. 第一阶段把其 `context_records` 作为证据，把原答案仅作为候选草稿；
2. 若某模式无法返回足够细粒度上下文，则保留原答案并明确标记为 `derived_answer`，统一答案生成器不得把它伪装成原文。

最终推荐第一种：统一智能体负责最后答案，保证两种策略具有一致的引用、拒答和风格。

GR 自动模式会结合当前知识范围选择查询方式：来源范围只有一篇文档，或问题明确点名一篇文档时，“总结全文”使用 Local，避免为单文档启动项目级社区报告 Map/Reduce；多文档全局问题仍使用 Global。显式 Basic/Local/Global 始终优先，不会被自动策略改写。Global 查询会依据项目模型的实际上下文窗口调整报告批次，只改变查询时打包大小，不修改不可变 Bundle、社区报告或 Prompt；例如 262144 Token 的当前模型将批次上下文限制为 65536 Token，并保留指令与输出空间。

## 10. 多路召回融合与重排

### 10.1 归一与去重

不能直接比较不同通道的原始相似度。先按通道保留排名，再以稳定来源键去重：

```text
dedup_key = project_id + snapshot_ref + source_type + source_record_id + span
```

同一原文被 Chunk、关系和社区报告重复召回时，合并 `retrieval_channels`，但保留关系或报告提供的结构语义。

社区导航命中的底层关系仍使用 `relation` 类型，只额外记录 `community_navigation` 召回通道；若它同时被普通图检索命中，则按稳定关系 ID 合并，不重复占用证据集合。

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
- 上下文长度只作为预算内的小幅成本惩罚，不能让短而无关的片段挤掉高相关的报告或关系；模型未产出可校验声明时必须失败关闭，禁止把排名第一的原文直接包装成答案。

检索计划会在两个位置执行：先对宽候选池按意图预留证据族名额，再在最终回答上下文中按实际可用槽位等比例缩放并再次预留。某个证据族没有足够候选时，空位按全局最终排名回填，因此动态计划不会因为图谱陈旧、社区稀疏或单一 Native 表而缩小实际召回。最终执行记录包含 `available_by_family`、`reserved_by_family`、`selected_by_family` 和 `backfilled`。

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

实现时另设一个保守的“联合弱匹配”硬门槛：仅当交叉编码器绝对相关度、Embedding 相似度和问题要点覆盖率同时低于校准值时，直接判为 `INSUFFICIENT`。RRF 是候选集内的相对排序，最佳候选即使完全无关也可能得到 1.0，因此不得用 RRF 单独证明答案存在。

充分性只回答“这些证据够不够”，不能代替“问题是否属于当前知识库”的范围判断。当前实现另有 `knowledge-agent-query-scope-v1` 门禁：

1. 对智能体身份、能力、使用方式和简单问候执行高精度全句交互意图识别，命中后不检索，只返回统一角色说明；
2. 普通知识问题完成召回与重排后，交叉编码器强相关或“Embedding 强相关且问题要点有覆盖”时直接进入回答；
3. 边界候选由项目 LLM 执行一次结构化 `knowledge / out_of_scope / unsupported / agent_meta` 分类。模型只读取问题、知识库元信息和有界证据摘录，不生成答案；
4. `out_of_scope`、`unsupported`、模型异常或非法输出均失败关闭，清除回答上下文中的选中标记，不调用事实回答模型，也不展示无关证据作为引用。

这样既避免用固定领域词表限定知识库，又避免 Embedding 对“系统、方法、作用、能做什么”等泛化表达产生的偶然近邻越过回答边界。

### 11.2 处理结果

- `SUFFICIENT`：生成答案；
- `RETRYABLE_GAP`：最多进行一次查询改写或补充通道；
- `AMBIGUOUS_SCOPE`：请求用户澄清；
- `CONFLICTING_EVIDENCE`：展示冲突来源，不擅自裁定；
- `INSUFFICIENT`：明确说明当前知识库没有足够依据。

当前实现禁止无限循环。首次证据为 `INSUFFICIENT` 或 `RETRYABLE_GAP` 时，最多执行一次针对缺口的补充计划：优先复用首次宽召回中尚未进入排名池的候选；确有必要时，KF 最多改写一次词面查询，GR 最多切换一次 Native 模式。补检索计划将 `retry_allowed=false`，因此第二次充分性判断后只能回答、限定回答或拒答，不会再次循环。

## 12. 答案生成与引用

### 12.1 生成契约

答案模型只接收：系统规则、用户问题、必要会话摘要、结构化证据和输出 Schema。文档文本始终标记为“不可信数据”，其中的指令不得改变系统行为。

输出结构：

```json
{
  "answer": "...",
  "answerability": "supported|partial|insufficient|conflict",
  "claims": [
    {
      "text": "...",
      "citations": ["E1"],
      "status": "supported|qualified|conflicting"
    }
  ],
  "limitations": [],
  "follow_up_questions": []
}
```

模型只看到本轮短证据键（`E1`、`E2`……），服务端在输出后把它们映射回不可伪造的 run-scoped `evidence_id`。这样既降低本地模型复制长哈希的格式失败率，也不放宽引用校验；未引用已给短键或引用未知键的声明仍会被删除。

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

## 14. LangGraph 与分层管控的边界

### 14.1 不把整条问答链机械迁入 LangGraph

知识问答的主体仍是固定有界流程：校验权限 → 问题理解 → 检索 → 融合 → 证据判断 → 生成 → 引用校验。需要持久状态、语义消歧和条件分支的部分交给 LangGraph；工具白名单、项目隔离、证据门禁和输出契约继续由确定性 Python 代码执行。这样既能获得状态图的可恢复性，又不会把数据库权限、引用完整性等硬约束交给模型决定。

### 14.2 从第二阶段开始使用的原因

当系统出现以下需求时引入 LangGraph：

- 根据问题动态选择多个检索分支；
- 证据不足时有限循环改写和补检索；
- 多轮会话状态和流程检查点；
- 人工确认知识库范围、敏感操作或高风险输出；
- 进程中断后从节点边界恢复；
- 查看每个节点的输入、输出、耗时和错误。

LangGraph 官方持久化机制以 `thread_id` 管理检查点，可用于故障恢复、人工介入和会话记忆；中断恢复可能重新执行节点，因此产生副作用的步骤仍必须幂等。Knowledge Forge 生产环境应使用 PostgreSQL 持久化 checkpointer，不使用内存 checkpointer。

### 14.3 当前问题路由子图

当前 `knowledge_forge/agent/context.py` 中的状态图为：

```text
START
  → input_rail
      ├─ agent_meta → agent_meta → END
      └─ knowledge → load_context → resolve_context → dialog_rail
                         ├─ clarify → clarify → END
                         └─ ready → finalize_input_route
                                      ├─ agent_meta → agent_meta → END
                                      ├─ out_of_scope → out_of_scope → END
                                      └─ knowledge → knowledge → END
```

- `input_rail` 只识别高置信度的智能体身份、能力和使用方式问题，避免为这类问题读取会话或检索知识库；
- `load_context` 按模型实际 Token 窗口加载最近完整轮次和滚动摘要；
- `resolve_context` 使用结构化 LLM 判断问题独立、指代已解析或存在歧义，历史内容只用于理解，不成为回答证据；
- `dialog_rail` 对歧义问题转入澄清，对可解析问题继续；
- `finalize_input_route` 使用补全后的独立问题，对照知识库名称、说明和文档范围执行结构化语义路由；只有高置信度知识库外问题才跳过检索，低置信度或模型异常继续检索并交由后置证据门禁复核。

状态图之后仍进入有界的问答主流程，不允许模型自由选择任意工具：

```text
authorize
  → question_control_graph
  → plan_retrieval
  → retrieve → fuse_and_rerank
  → scope_and_evidence_gate
      ├─ agent_meta / out_of_scope / unsupported → policy_response
      ├─ insufficient → controlled_refusal
      └─ sufficient / qualified → grounded_answer
  → citation_and_output_gate
  → persist_and_stream
```

### 14.4 六层策略协议

每轮运行将决策写入 `agent_runs.metadata.policy_control`，协议版本为 `knowledge-agent-layered-policy-v1`。这些记录是可审计决策，不包含模型隐藏思维链。

| 层 | 决策依据 | 可执行动作 | 硬约束 |
|---|---|---|---|
| 访问层 | 会话能力、项目、构建策略 | allow / deny | 项目与策略隔离；未授权不能进入检索 |
| 输入层 | 当前问题、独立问题与知识库范围资料 | allow / route | 智能体元问题和高置信度越界问题不进入知识检索；不确定时保留召回 |
| 对话层 | 历史状态和结构化语境解析 | allow / clarify / skip | 歧义时先澄清，不猜测指代 |
| 检索层 | 问题范围、来源有效性和证据充分性 | allow / qualify / refuse / skip | 越界或证据不足不生成知识声明 |
| 执行层 | 当前构建策略的工具白名单 | allow / deny / skip | KF 与 GR 检索器不能串用；工具均为当前只读问答能力 |
| 输出层 | 回答类型、声明和引用 | allow / deny | 知识声明必须有通过服务端核验的引用；策略响应不得夹带无依据事实 |

语义与确定性职责保持分开：LLM 负责多轮指代解析、检索前知识范围分类以及边界证据的语义对齐；代码负责高置信度门槛、失败降级、权限、工具范围、调用上限、快照新鲜度、来源有效性、引用映射和最终发布。检索计划当前采用可审计的确定性分类与预算，不依赖模型自由规划；后续只有离线评估证明收益时才增加受约束的模型规划分支。

### 14.5 建议的完整状态

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

### 14.6 后续多知识库阶段的建议节点

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

当前单知识库接口：

```text
GET    /api/agent/knowledge-bases
GET    /api/agent/threads?project_id={project_id}
POST   /api/agent/threads
GET    /api/agent/threads/{thread_id}
DELETE /api/agent/threads/{thread_id}
POST   /api/agent/threads/{thread_id}/messages
GET    /api/agent/runs/{run_id}
GET    /api/agent/runs/{run_id}/trace       # 安全的执行过程投影
POST   /api/agent/runs/{run_id}/cancel
GET    /api/agent/runs/{run_id}/events      # SSE
POST   /api/agent/runs/{run_id}/feedback
GET    /api/agent/evidence/{evidence_id}
```

创建会话时由服务端校验并固化一个知识库：

```json
{
  "project_id": 1
}
```

提交消息时使用当前线程的知识库；`mode` 只对 GraphRAG Native 生效，Knowledge Forge 固定使用自身的动态混合检索：

```json
{
  "query": "...",
  "mode": "auto|basic|local|global"
}
```

第三版显式多选不能直接把 `project_ids` 塞入现有消息请求。它需要新增版本化请求合同、逐库授权和快照集合，并保留现有单知识库线程的兼容语义。

SSE 事件只发送阶段、增量答案和安全元数据，不泄露隐藏 Prompt、数据库凭据或完整模型原始响应。`/trace` 从已授权运行的 `agent_run_steps`、`agent_tool_calls` 和运行摘要构造轻量视图；浏览器运行中轮询该接口，历史轮次只在用户展开时读取。接口不返回原始请求/响应载荷、模型 scratchpad、文件路径或候选答案。

## 17. Web 产品设计

### 17.1 第一版

- 项目页增加“知识问答”；
- 默认固定当前项目；
- 展示流式回答、引用卡片、页码、章节和媒体；
- 点击引用打开原文片段，图片和表格可放大；
- 显示“本次使用：知识块/实体关系/社区报告”；
- 每轮回答提供可折叠执行过程，实时展示安全的处理摘要与实际工具调用，刷新后可恢复；
- 提供停止生成、复制、反馈和重新检索。

### 17.2 后续版本

- 多知识库选择器与自动选择解释；
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
- KF 在多社区或首轮关系稀疏时，允许执行一次有界社区导航召回，但不形成工具循环；
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

完成标准：服务重启后可恢复会话和已完成轮次，未完成模型输出明确失败关闭；追问评估优于无会话基线；工具调用次数受预算约束。

当前进度（2026-08-31）：会话、消息、来源恢复和 LangGraph `input_rail → load_context → resolve_context → dialog_rail → knowledge/agent_meta/clarify` 问题控制子图已经落地；官方 PostgreSQL checkpointer 保存有界最近消息、滚动摘要、独立检索问题和流程位置。历史回答只用于指代解析，不进入 `EvidenceItem`。问答主流程增加访问、输入、对话、检索、执行和输出六层策略协议：智能体元问题提前分流，歧义问题先澄清，KF/GR 工具按策略白名单隔离，越界或无直接支持的问题受控拒答，知识声明必须通过服务端引用门禁。每轮决策连同确定性问题分类、KF/GR 动态检索计划、候选池与最终上下文双层证据族配额、编号图表精确恢复、GR 单文档/全局成本路由和至多一次补检索一起进入运行审计。用户可展开查看安全投影后的阶段、工具和具体证据，但不会看到模型隐藏思维链。已完成运行和会话可在刷新或重启后恢复；服务中断时正在生成的回答会明确失败关闭，不会盲目续写旧模型输出。第二版的单知识库范围已完成，自动多知识库选择和复杂问题分解归入第三版。

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
│   ├── planning.py           # 问题形态与动态检索计划
│   ├── retrieval/
│   │   ├── knowledge_forge.py
│   │   ├── graphrag_native.py
│   │   └── source_context.py
│   ├── fusion.py             # 去重、RRF、rerank、预算选择
│   ├── sufficiency.py        # 充分性和冲突
│   ├── scope.py              # 智能体意图与知识库回答范围门禁
│   ├── policy.py             # 六层策略决策、工具白名单与输出门禁
│   ├── answer.py             # 结构化答案生成
│   ├── citations.py          # 确定性引用校验
│   ├── context.py            # LangGraph 多轮语境与上下文压缩
│   ├── trace.py              # 面向用户的安全执行审计投影
│   └── workflow.py           # 有界问答流程与动态分支
├── database.py               # 仅持久化，不放智能体决策
├── service.py                # 应用编排入口
└── web.py                    # HTTP/SSE 与展示
```

智能体模块只能通过现有服务/数据库的受控查询接口读取知识，不应复制一套知识构建逻辑。

## 23. 实施顺序与当前建议

阶段 0、第一版和第二版的单知识库范围已经落地。下一轮应从第三版的“显式多选知识库”开始，先建立跨库授权、快照隔离、并行检索和冲突展示，再用评估结果决定是否开放自动知识库选择与 DRIFT。不要直接引入多智能体或无界工具循环。

后续重点验证：

- 自动知识库选择是否比用户显式选择更有价值；
- 多文档全局问题使用社区报告或 GR Global 的质量/延迟收益；
- 一次补充检索在开放问题集上的 Recall/Precision 净收益；
- 动态证据族预算在不同领域、语言与文档结构上的稳定性；
- DRIFT 是否能在严格成本上限内显著改善复杂探索问题。

每个新增节点都必须证明质量收益或可靠性价值，否则保持简单流程。

## 24. 已实现状态

阶段 0 与第一版已经按本路线落地：

- 统一 `SnapshotRef`、`SourceRef` 和 `EvidenceItem`；证据 ID 按运行隔离，稳定键绑定项目、快照、记录和内容摘要；
- KF 适配器只读取当前来源，图谱不新鲜时只使用当前 revision 的 Chunk 与向量；GR 适配器只读取当前 READY Native Bundle，并同时核对 run、source snapshot 和运行清单摘要；
- 实现多通道加权 RRF、独立 HTTP cross-encoder reranker、确定性降级排序、类型/文档配额和 Token 预算选择；
- 实现证据充分性、冲突和歧义状态；联合交叉编码器绝对相关度、Embedding 相似度和问题要点覆盖率识别无答案，确定拒答时不调用 LLM；
- 答案只执行一次有界结构化生成；模型使用 `E1/E2` 短证据键，服务端映射回运行级证据 ID；GraphRAG 原回答仅作为不可引用草稿；服务器删除未知、无引用或非法声明，并重新生成引用编号；
- PostgreSQL 新增会话、运行、消息、步骤、工具调用、证据、声明引用、SSE 事件和反馈账本，外键约束保证引用与证据属于同一运行；
- 实现显式单知识库 HTTP API、能力会话隔离、可重放 SSE、安全停止、来源详情和反馈；
- 实现真实问答工作台，支持 KF 与 GR 模式差异、历史消息、停止、复制、重新检索、来源文件/页码/图片/表格查看；
- 实现每轮可折叠执行过程：轻量授权接口只投影阶段、决策、工具名称、状态、计数和耗时；每个步骤或工具可继续展开查看白名单化的输入范围与结果摘要。证据相关阶段额外读取至多 24 条已选证据的 500 字有界摘要，显示文件、页码/章节并可进入独立授权详情；生成依据与最终引用明确区分，完整证据正文和消息记录不进入轮询接口。运行中实时更新，历史展开后按需恢复，原始 Prompt、工具载荷和隐藏思维链不进入浏览器；KF 不展示共享计划对象中的 GraphRAG Native 查询方式，Basic/Local/Global 只对 GR 知识库可见；
- 固化 28 题中英文工程校准集（含 4 题无答案探针）及真实 HTTP 评估脚本；
- 第一版问答主体保持普通 Python 固定编排；第二版已为多轮语境和问题分支加入 LangGraph，但没有把检索、数据库访问和发布门禁改成模型自主工具循环。当前仍未加入自动多知识库选择、DRIFT 工具循环或长期记忆。
- 2026-08-26 增加 Knowledge Forge 社区导航二级召回：事实桥和内容相近关联只负责选路，最终候选必须还原为带原文来源的已验证关系；普通局部查询在首轮关系充分时不执行扩展。
- 2026-08-28 完成 `knowledge-agent-retrieval-plan-v1` 动态检索：按事实、实体、关系、过程、比较、全局、来源和图表问题调整文本/实体/关系/社区预算；候选池和最终上下文均执行可缩放配额并自动回填。KF 对明确编号的图表增加当前 revision 内的精确恢复；证据不足时最多执行一次补充计划。GR 自动模式将单文档总结路由到 Local，多文档全局总结保留 Global，显式模式不被覆盖；Global 按实际模型上下文进行查询批次打包。

实现代码以 `knowledge_forge/agent/` 为边界；`0035_knowledge_agent_v1` 建立会话、运行、消息、步骤、工具调用、证据、引用、SSE 事件和反馈账本，`0036_knowledge_agent_context_runtime` 为官方 PostgreSQL checkpointer 建立隔离的 `knowledge_agent` schema。2026-08-24 在冻结的 28 题双语工程校准集上通过全部门禁：任务成功率、引用解析率、声明引用完整率和无答案拒答率均为 100%，答案要点召回率为 92.43%，跨项目证据为 0；同时完成 KF、GraphRAG Native、SSE、停止、反馈与来源详情的真实 HTTP/模型链路验证。该结果用于工程回归，不代表开放领域通用质量结论。

2026-08-26 的社区导航增量通过完整测试集（1347 passed，45 项按环境跳过），并通过真实公共 HTTP 问答入口验证：运行在当前 KF 快照上触发多社区导航，选中的内容关联最终解析为 2 条带来源的已验证关系，`synthetic_evidence_count=0`，reranker 正常过滤未进入最终上下文的弱相关扩展。随后在正确的 10 文档冻结语料上重跑全部 28 题，任务、引用解析、声明引用完整、无答案拒答仍为 100%，答案要点召回率仍为 92.43%，跨项目证据为 0，全部门禁通过。该结果证明当前基准未回退；社区导航对更广开放问题集的 Recall/Precision 净收益仍应通过后续配对评估持续校准。

2026-08-28 开始第二版的多轮语境增量：新增 `knowledge_forge/agent/context.py`，以 LangGraph `StateGraph` 编排读取上下文、结构化解析和条件分支；以会话 `thread_id` 连接官方 PostgreSQL checkpointer，并在独立 `knowledge_agent` schema 中保存执行状态。上下文已从固定“最近 8 条”升级为本地模型 Token 预算窗口：先为系统提示、Schema、当前问题、输出、安全余量以及结构化输出修复请求预留 Token，再动态保留最近完整轮次，较早轮次按水位压缩到受 Token 上限约束的滚动摘要；消息数量和字符数仅作为极端输入保护。语境请求不再固定使用 8192 Token：系统先交叉验证知识库配置与 Ollama 当前 runner 上下文，再按完整提示需求以 4096 Token 为阶梯在 8192～32768 Token 范围内扩容，同时默认不超过实际模型窗口的 50%；完整历史能放入动态窗口时保持原文，只有超过动态上限或极端消息保护线时才压缩。每轮将项目配置上限、runner 有效上限、动态天花板、估算需求、最终预算与 `minimum/expanded/capped` 原因写入运行元数据。失败或取消的轮次不会进入语义上下文。第一轮独立问题不调用语境解析模型；当知识库范围资料可用时，检索前最多调用一次 qwen3.6 结构化范围分类。常规后续轮次最多另调用一次结构化语境解析；本地模型常见字段别名由代码归一，单数指代被解析为多个候选时由确定性规则改走澄清分支。独立问题用于范围路由、检索和 rerank，原问题继续用于展示与审计，历史文本不成为回答证据。

同日的真实公共 HTTP/Ollama 多轮验收覆盖 KF 与 GR：KF 连续四轮正确处理定义、加密共享机制、“它”指向加密共享的比较问题，以及“图1”A/B/C 精确视觉问题，最终证据同时保留文本、实体、关系、页码和图片；一个带不存在设备编号的问题触发一次候选池补充后确定性拒答。GR 自动模式的单文档全文总结以 Local 完成，后续文档外数量问题正确补全文件范围并回答“未给出”；显式 Global 保持 Global，以 65536 Token 查询批次在 116 秒完成，未再触发旧路径的 300 秒超时。上述结果用于当前两份真实文档的工程验收，不替代跨领域基准评估。

2026-08-31 完成 `knowledge-agent-layered-policy-v1`：LangGraph 问题控制子图增加输入预检、语境门禁和 `knowledge / agent_meta / clarify` 条件路由；问答主流程增加访问、检索、执行和输出策略检查。全量回归为 1406 passed、45 skipped。真实公共 HTTP/Ollama 验收覆盖 Knowledge Forge 能力问题提前分流、知识问题有引用回答、同会话“它”补全为独立问题、文档外烹饪问题受控拒答，以及 GraphRAG Native Local 有引用回答；KF 与 GR 只调用各自允许的检索器。每轮持久化六层决策，执行过程可见其安全摘要，知识声明仍由服务端引用校验决定是否发布。

2026-09-01 将当前维护文档统一到上述实现，并重新执行完整回归：1412 passed、45 skipped，Python 编译与文档本地链接/格式检查通过。这里新增的 6 项测试来自当前工作树中后续补充的回归覆盖，不改写 2026-08-31 分层管控验收当时的历史计数。

## 25. 参考依据

- LangGraph Persistence：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Interrupts：<https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph Memory：<https://docs.langchain.com/oss/python/langgraph/add-memory>
- Microsoft GraphRAG Query Overview：<https://microsoft.github.io/graphrag/query/overview/>
