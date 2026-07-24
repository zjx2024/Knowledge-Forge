# Knowledge Forge：关系抽取与验证闭环 P0 改造任务

## 你的角色

你正在维护仓库 `zjx2024/Knowledge-Forge`。请作为资深 Python、PostgreSQL 与知识图谱工程师，直接完成本任务中的代码修改、数据库兼容改造、自动化测试和自审。

这是一次工程实现任务，不是只做分析或给建议。先检查仓库并给出一个简短执行计划，随后无需等待确认，直接实施；只有当缺少必要权限、必需凭据或存在会改变产品语义的重大歧义时才暂停提问。

## 开始前必须做的事

1. 阅读仓库根目录 `AGENTS.md`、`README.md`、`docs/PROJECT_STATUS.md`、`docs/IMPLEMENTATION_PLAN.md`，遵守其中的架构约束。
2. 检查 `git status --short`。保留用户已有修改，不覆盖、不回滚、不清理无关文件。
3. 用 `rg` 定位下列实现及其测试：
   - `knowledge_forge/schema_evolution.py`
   - `knowledge_forge/chunking.py`
   - `knowledge_forge/service.py`
   - `knowledge_forge/database.py`
   - `knowledge_forge/hugegraph_store.py`
   - `knowledge_forge/entity_resolution.py`
   - `knowledge_forge/entity_identity.py`
4. 从仓库配置、`AGENTS.md`、README、测试配置和 CI 中确认真正的安装、格式化、静态检查和测试命令。不要臆造命令。
5. 在修改前运行与抽取、关系复核、数据库和 HugeGraph 同步有关的现有快速测试，记录基线结果。如果环境缺少 PostgreSQL、HugeGraph、Ollama 或模型，不要安装大型服务，也不要伪造成功；优先使用现有 mock/fake，并在最终报告中精确说明未运行项和原因。

## 背景与已确认根因

当前链路已经能完成上传、解析、切块、抽取、归一、入库和 HugeGraph 同步，但关系质量闭环存在系统性缺陷：

- 复核结果使用数组下标关联候选，应用层没有验证返回是否完整、唯一、合法。
- 复核器返回空数组、遗漏候选或调用失败时，关系仍可能通过宽松回退进入数据库。
- “两个端点在同一句中出现”被当成可接受证据，但这只能证明共现，不能证明谓词、方向、否定或因果语义。
- 复核协议只能修正关系类型，不能交换 source/target 或修正方向。
- 不兼容关系会被静默改成 `related_to`，从而隐藏抽取错误。
- `evidence_unverified` 关系会进入 HugeGraph，污染默认图遍历。
- 抽取使用了带 `章节ID`、`来源文档`、`分块` 等合成头部的完整 Chunk，容易把结构元数据误抽成实体。
- 部分置信度写入使用 `value or 0.5`，会把合法的 `0` 错改成 `0.5`。
- 服务日志只统计通过/拒绝，无法区分漏答、调用失败、重试及隔离。

不要把“换模型”或“调低/调高阈值”当成本任务的主要修复。先修复候选—复核—发布协议。

## 本轮目标

实现一个 fail-closed、可审计、可测试的关系复核与发布闭环：

```text
干净正文 -> 关系候选 -> 严格复核 -> 已接受关系 -> PostgreSQL / HugeGraph
                         -> 漏答、失败、不确定 -> 审核/隔离区，不发布
```

本轮只完成 P0。不要顺手重写整个抽取器、替换模型、引入大型新依赖、实现标注 UI 或全面重构实体类型体系。可以为后续 P1 留出清晰接口，但必须保持改动可审查。

## 必须实现的修改

### 1. 稳定候选标识与严格复核协议

为每个待复核关系生成稳定的 `candidate_id`，不得再依赖返回数组的序号作为唯一关联依据。

- 标识应在一次抽取/重试中稳定，可使用确定性哈希或项目现有 UUID 约定。
- 哈希输入至少包含 chunk、source mention、target mention、proposed relation type 和候选序号/证据定位，以避免碰撞。
- 复核请求和响应都显式携带 `candidate_id`。

复核响应至少支持：

- `candidate_id`
- `verdict`: `supported | contradicted | insufficient_evidence`
- `reason`
- `corrected_type`（可选）
- `orientation`: `as_is | reversed`，或等价的修正 source/target ID 机制
- 精确证据文本；若项目已有可靠偏移体系，也保存 evidence offsets

优先使用 Ollama JSON Schema / structured output 与项目现有 Pydantic 模式进行结构校验；temperature 设为适合确定性判定的低值。不要只在 Prompt 中要求 JSON 后直接信任任意字典。

应用层必须验证：

- 每个请求候选恰好对应一个合法响应；
- 不得出现未知 ID、重复 ID或缺少 ID；
- verdict、关系类型、方向和端点修正均属于允许集合；
- `supported` 必须提供能在允许上下文中定位的证据，并能支持修正后的端点、谓词和方向；
- `contradicted` 与 `insufficient_evidence` 不得发布。

若首轮响应漏掉部分候选，只对漏项以小批次、低温重试一次。仍漏答则标为 `verification_missing`。调用异常、超时、解析失败或非法输出标为 `verification_failed`。这两种情况都必须 fail closed，不能因自报置信度或端点共现而发布。

### 2. 明确状态机与发布白名单

集中定义关系验证状态，避免各模块使用散落字符串。至少包括：

- 可发布：`rules_verified`、`llm_verified`、`manual_verified`
- 不可发布：`rejected`、`uncertain`、`verification_failed`、`verification_missing`

如果需要兼容已有 `evidence_unverified` 数据，可以继续读取/展示，但它必须被视为不可发布状态。

发布逻辑必须使用白名单，而不是“不是 rejected 就发布”：

- PostgreSQL 可以保存不可发布候选/关系，供审计和后续人工处理；
- HugeGraph 同步只能读取可发布状态；
- 不要通过全局修改所有关系查询而意外隐藏管理页面所需数据。新增明确的 publishable 查询或参数，并只在图发布路径使用；
- HugeGraph 边应保留适用的 verification status 和 provenance/evidence 属性，或至少确保同步查询严格过滤；
- 重同步、重入库、删除文档和多文档贡献回收仍须满足现有幂等与隔离约束。

### 3. 收紧规则验证

删除“端点同句共现 + 模型自报高置信度即可 `rules_verified`”的逻辑。

`rules_verified` 只能用于代码可以确定验证的有限词法/结构规则，并同时确认：

- 明确谓词表达；
- source/target 方向；
- 没有命中否定；
- 类型/domain/range 兼容。

如果当前代码没有足够可靠的确定性规则，宁可把候选送 LLM 复核或标为 `uncertain`，不要为了维持关系数量继续宽松通过。

停止将不兼容关系静默降级为 `related_to`。应保留原始候选及拒绝/不确定原因，或送入严格复核；不要制造语义上不同的新关系。

### 4. 从干净正文抽取

`chunking.py` 会为存储/展示拼接 `# title`、`章节ID`、`来源文档`、`分块` 等合成头部。抽取实体和关系时应使用干净正文，例如复用并完善现有 `_chunk_body`，而不是把合成头部作为可抽取文本。

- 标题、文档路径、章节信息可以作为单独的只读上下文提示，但必须明确标记为不可抽取元数据；
- evidence 的定位和保存必须仍然与项目使用的正文/Chunk 定位约定一致；
- 不改变 Chunk 的持久化内容和展示格式，除非现有测试证明必须调整。

### 5. 候选与复核审计

按仓库现有数据库风格进行向后兼容、追加式改造。不要删除数据或使用破坏性迁移。

新增或等价实现以下审计能力：

#### `relation_candidates`

保存候选的稳定 ID、project/document/chunk 范围、原始 source/target、原始关系类型、原始证据、模型自报分数、当前状态、创建/更新时间等。

#### `relation_verification_events`

以 append-only 方式保存每次复核事件：candidate ID、attempt、model、prompt version、schema version、请求/原始响应（注意避免记录密钥）、解析后的 verdict、reason、错误类别、时间戳。

具体列名和 ID 类型应适配项目现有数据库约定。要求：

- SQL 幂等、可在已有数据库上安全升级；
- 所有派生数据保留 `project_id`，需要图隔离处保留 `project_key`；
- 不在启动过程中删除旧列或旧数据；
- 若仓库已有 migration 机制则使用它；若没有，采用项目当前的幂等启动 DDL 风格，并在最终报告中指出后续引入版本化 migration 的建议；
- 为常用 project/document/chunk/candidate/status 查询添加必要索引；
- 原始 JSON 使用适当的 JSONB/现有序列化约定。

### 6. 可观测性与兼容性修复

任务日志和结果摘要至少统计：

- verification requested
- rules verified
- LLM verified
- rejected
- insufficient/uncertain
- missing decisions
- verifier failures
- retry count
- quarantined/not published
- published to HugeGraph

日志中不得输出数据库密码、模型服务凭据或完整敏感文档。

修复相关持久化代码中的 `confidence or 0.5` 问题：仅当值为 `None`/缺失时使用默认值，必须保留合法的 `0`。搜索同类写法并只修复语义相同的真实问题。

## 必须增加的自动化测试

测试名称与组织遵循仓库现有规范。尽量使用纯单元测试和 fake verifier，使测试不依赖真实 Ollama、PostgreSQL 或 HugeGraph。至少覆盖：

1. 复核返回空数组：所有候选均不可发布，重试后状态为 missing/failed。
2. 复核漏一个 `candidate_id`：只重试漏项；最终仍漏时该候选不可发布。
3. 复核返回重复 ID：视为协议错误，不得发布受影响候选。
4. 复核返回未知 ID：视为协议错误，不得错误关联其他候选。
5. 复核调用抛异常、超时或返回非法 JSON：fail closed，记录 failure。
6. 明确 `supported` 且证据合法：进入 `llm_verified`。
7. `contradicted` 与 `insufficient_evidence`：均不可发布，并保留原因。
8. `orientation=reversed`：正确交换端点，同时保留原候选审计记录。
9. 两端点同句出现但谓词/方向不成立：不得仅因共现而 `rules_verified`。
10. 不兼容谓词：不得静默变成 `related_to`。
11. 合成 Chunk 头部中的文件名、章节 ID、页码或“分块”字段不会成为实体；正文实体仍能抽取且证据可定位。
12. HugeGraph 同步只接收 `rules_verified | llm_verified | manual_verified`；`evidence_unverified`、missing、failed、uncertain、rejected 全部不进入图。
13. confidence 为 `0` 时数据库写入仍为 `0`。
14. 多项目/多文档隔离、重入库和删除贡献的现有关键测试不回归。

如数据库测试基础设施允许，再增加候选与事件表的幂等 DDL、外键/索引、append-only 事件及发布查询测试。不要因为集成环境不可用而删除测试；可以清楚标记需要服务的测试，并运行其余测试。

## 验收标准

只有同时满足以下条件才能声明完成：

- 复核器漏答、报错或返回非法数据时，没有任何受影响关系进入可发布状态；
- `evidence_unverified` 不再同步到 HugeGraph；
- 关系方向可由复核器明确纠正并被测试覆盖；
- 不再使用端点共现和 LLM 自报高分作为充分语义证据；
- 不兼容谓词不再静默降级成 `related_to`；
- 合成 Chunk 头部不参与实体/关系抽取；
- 候选及每次复核事件均可审计；
- 合法 confidence `0` 不被改写；
- 新增测试全部通过，现有相关测试不回归；
- 代码格式、静态检查和项目已有质量门禁通过；
- 没有引入未经需要的大型依赖，也没有改变 PostgreSQL 为主存储、HugeGraph 为遍历图的既有职责。

## 验证执行顺序

修改完成后必须按以下顺序验证并修复发现的问题：

1. 运行新增的最小单元测试集合。
2. 运行抽取、schema evolution、数据库、service、HugeGraph、删除/重入库相关测试。
3. 运行项目完整测试套件。
4. 运行项目已有 formatter/linter/type checker；只格式化本次涉及文件，不制造无关大 diff。
5. 检查 `git diff --check`。
6. 阅读完整 diff，做一次面向以下风险的自审：fail-open、状态白名单遗漏、跨项目数据泄漏、SQL 非幂等、重试重复写、方向纠正后证据错位、删除贡献回归、敏感日志。
7. 如果仓库有可用的本地轻量样例和必要服务，再做一次小型 smoke test；不要自动下载大型模型，也不要修改真实用户数据。

所有失败都要先判断是本次回归、已有失败还是环境阻塞。本次回归必须修复；已有失败不得通过删测试或放宽断言来掩盖。

## 禁止事项

- 不得执行 `git reset --hard`、`git clean -fd`、删除数据库/卷等破坏性命令。
- 不得覆盖或回滚用户已有修改。
- 未经要求不要 commit、push、创建 PR、部署或修改外部服务。
- 不得把测试改成无意义断言来“通过”。
- 不得把 verifier 失败默认当作通过。
- 不得仅靠提高/降低阈值宣称问题已解决。
- 不得在日志、fixture 或文档中写入真实凭据。
- 不要在本轮混入 Web 认证、CSRF、上传格式、实体类型大重构、向量索引等无关改造；这些应单独实施。

## 最终交付格式

完成后用中文给出简洁但可核验的报告：

1. **结果**：本轮是否达到验收标准。
2. **修改摘要**：按文件说明关键变化及设计决定。
3. **状态流转**：列出哪些状态可发布、哪些被隔离。
4. **数据库变化**：新增表/列/索引及兼容策略。
5. **验证证据**：逐条列出实际运行的命令、通过/失败数量；不要只说“测试通过”。
6. **未验证项**：缺失服务或数据导致未运行的内容及复现命令。
7. **风险与后续**：只列仍真实存在的风险，不夸大完成度。
8. **变更范围**：附 `git status --short` 和 diff 统计；不要提交或推送。

如果无法完成某项，必须说明具体阻塞、已经尝试的安全替代方案，以及当前代码是否仍处于安全的 fail-closed 状态。

## 本轮之后的路线（本轮不要实现）

P0 完成并稳定后，再单独规划：

- P1：实体 span/ID 固定后的二阶段关系分类，显式 `NO_RELATION`、候选剪枝、覆盖率驱动 gleaning、多句窗口、predicate/domain/range/inverse/symmetric 约束。
- P2：span-first NER、结构节点过滤、类型断言与 canonical identity 解耦、跨类型同名召回、cannot-link、可逆 merge/split、hard negatives。
- P3：50–100 个代表 Chunk 起步的金标准，扩展至 200–300；实体 exact-span F1、类型准确率、实体归一 pairwise F1/B³、严格关系 F1、证据覆盖率、false-merge rate、Brier/ECE 校准。
