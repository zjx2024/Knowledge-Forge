# Knowledge Forge P0.1：关系隔离与审计复核修复

## 任务性质

请在仓库 `zjx2024/Knowledge-Forge` 最新 `main`（至少包含提交 `b9b808fb3ece5ecb6ae0892e30ae150278b6d938`）上完成一次针对性修复。

这是 P0 复核后的补丁任务。保留现有严格 verifier、稳定请求内 `candidate_id`、漏项重试和 HugeGraph 发布白名单；不要回滚或重写已经通过的部分。先检查 `AGENTS.md` 和 `git status --short`，给出简短计划后直接实施、测试和自审。保留用户已有修改，不要 commit、push 或创建 PR。

## 已确认问题

### 1. 默认用户检索仍会返回隔离关系

当前所有关系都会写入 `relation_assertions`，这是允许的审计设计；但以下默认用户消费路径没有验证状态过滤：

- `Database.search_graph_text()`
- `Database.get_relations_by_ids()`
- `Database.list_project_relations()`
- `Database.statistics()` 中的关系数量
- `Service.search()` 会将上述结果直接返回 `/api/search`

因此 `rejected`、`uncertain`、`verification_failed`、`verification_missing` 等关系仍可出现在搜索结果和首页关系统计中。这违反“隔离但不发布”的语义。

### 2. 旧数据升级没有隔离弱规则结果

旧版本的 `rules_verified` 只检查端点共现、类型兼容和模型自报高分，不能证明谓词及方向。新版本却把所有 `rules_verified` 直接纳入发布白名单；同时没有处理旧 relation vectors 和已经存在的 HugeGraph 边。

### 3. evidence 字符偏移使用了错误坐标系

抽取现在基于去掉合成头部后的正文，但 `evidence_start/evidence_end` 被原样写入与完整 `chunks.content` 关联的记录。测试还把正文中的 `Alpha` 断言为 offset 0。正确契约应保证：

```python
chunk_content[evidence_start:evidence_end] == evidence_text
```

### 4. 未验证的 aliases 可以绕过端点证据检查

实体 aliases 目前由抽取模型自由返回，没有逐个验证其是否真实出现在正文。supported relation 的端点检查又接受任意 alias，因此虚构 canonical name 加一个正文常见词 alias 可能通过本地检查。

## 必须实现

### A. 统一所有默认消费端的发布白名单

建立一个集中、可复用的 publishable 查询约束，避免各方法各写一套状态字符串。

- 默认用户搜索、关系列表、关系统计、通过 ID 回取关系，只能返回：
  - `rules_verified`
  - `llm_verified`
  - `manual_verified`
- `Service.search()` 的 lexical 与 vector 两条路径都必须再次做数据库白名单过滤；不能信任“只有已发布关系才有向量”这一历史假设。
- 为审核后台保留显式命名的 all-status/candidate 查询，调用者必须主动选择，不能让默认产品查询返回隔离项。
- 首页“关系”数字改为已发布关系数；若需要展示候选总数，使用单独且明确标注的字段。
- 返回关系时可附 `verification_status`，便于诊断，但不能依靠前端隐藏完成安全过滤。

### B. 处理历史状态、向量和 HugeGraph 边

增加明确的 verification policy version，例如 `relation-verifier-p0-v1`，写入新候选/断言 metadata。

- 对缺少新策略版本的旧 `rules_verified`，迁移为不可发布的 `legacy_unverified` 或等价状态。不要把旧弱规则结果继续当作可信。
- 旧 `evidence_unverified`、`unverified` 及其他非白名单状态继续不可发布。
- 安全、幂等地软删除/停用这些关系对应的 relation vector embeddings。
- 当一个现有关系从可发布状态降级为不可发布状态时，必须同步停用旧向量；不能只是不再 upsert。
- 提供显式维护命令、服务方法或持久任务，用于对现有项目执行 HugeGraph 文档 purge + 白名单重同步。不要在应用启动时静默执行耗时的外部图重建，也不要在测试中修改真实用户图。
- 文档说明升级后的必需操作。仅调用 `sync_document_assertions()` 不足以证明旧边已删除；重同步流程必须先清除该文档旧关系边再写入已发布边。

如果无法自动区分旧规则和新规则，采取保守策略：旧数据一律降级并重新复核，不得猜测其可信。

### C. 修正正文与完整 Chunk 的 evidence 坐标

将正文解析改为返回正文及其在完整 `chunks.content` 中的起始偏移，例如：

```python
body, body_start = _chunk_body_with_offset(chunk)
```

要求：

- 只识别项目真实生成的完整合成头格式；正文中自然出现一行“分块:”时不得误截断。
- 如果调用 `.strip()`，必须把被去掉的前导字符数计入 `body_start`。
- LLM 抽取和 verifier 仍只看干净正文。
- 在最终返回/持久化前，将实体、关系及候选的 evidence offsets 转换成完整 `chunks.content` 坐标。
- `_original_candidate` 与纠正后的关系使用同一明确坐标系。
- exact quote 重复出现时，优先保留/验证候选原位置附近的匹配，不要无条件改成全文第一次出现。

### D. 收紧实体 surface form 与关系端点证据

在不全面实施 P2 span-first NER 的前提下，先做 P0 安全修复：

- 不能仅因为任意 `evidence` 字符串可定位，就接受一个 name/alias 均未在正文出现的实体。
- 对 name 和每个 alias 分别做来源定位；只保留在正文有 exact source span 的 surface forms，或保存明确的 `verified_surface_forms`。
- supported relation 的两端点检查只能使用这些经过来源验证的 surface forms。
- 对短 ASCII 名称/缩写使用合理的字符边界，避免 `AI` 命中 `claim`、`US` 命中 `uses`。
- 原始、未验证 aliases 可以留在 raw audit metadata，但不能参与自动发布判定。

### E. 修正方向/类型判定的残余一致性问题

- 当 `orientation=reversed` 或 `corrected_type` 改变时，不能继续发布描述原错误方向的旧 `description`。可以清空该描述、生成保守的规范三元组描述，或让严格 schema 返回并验证 corrected description。
- `contradicted` 和 `insufficient_evidence` 应先形成不可发布 verdict；不要因为原始 proposed type 不在 Schema 且未给 corrected type，就把合理的拒绝错误记为 verifier failure。
- Prompt 与应用校验必须一致：不要一边允许 verifier 保留 proposed type，另一边又因该类型不在 Schema 失败。

### F. 审计保真与持久化一致性

- 修复通用 JSON 序列化中 `value or {}` 把 `[]`、`0`、`false`、空字符串改成 `{}` 的问题；只有 `None` 才使用默认对象。确认不会破坏现有调用。
- candidate、verification events、relation assertion 的数据库写入应在一个 Chunk 级事务内保持一致。至少不能出现 candidate 已标 `llm_verified`，但事件或 assertion 因后续数据库错误缺失的半完成状态。
- 向量和 HugeGraph 属于外部副作用，可使用待发布/outbox 状态或在事务提交后执行并可重试；不要伪造跨系统原子事务。
- 保留 append-only 事件语义，并增加可区分重复入库的 `ingestion_run_id`、job ID 或 content version（采用项目已有标识）。

## 必须增加的测试

1. 将每个可发布与不可发布状态真实写入关系层，经 `Service.search()` 验证只返回白名单关系；同时覆盖 lexical 与 vector ID 回取。
2. `statistics()` 与默认项目关系列表只统计/返回已发布关系；审核查询仍能读取隔离候选。
3. 从旧版本状态升级：旧 `rules_verified` 被降级，旧非发布关系向量被停用，升级重复执行幂等。
4. 可发布关系降级为 `uncertain` 后，其旧向量不再被检索；图重同步后旧边被移除。
5. 合成头部 Chunk 中，实体和关系均满足 `full_content[start:end] == evidence_text`；覆盖多空行与正文自然出现“分块:”的情况。
6. 同一 exact quote 出现两次时，验证偏移不会无条件跳到第一次。
7. 虚构 name + 正文常见词 alias 不能使 supported relation 发布。
8. `AI` 不得在 `claim` 中被当作端点，`US` 不得在 `uses` 中被当作端点。
9. 方向反转后，持久化端点、relation type、description、vector 文本和 HugeGraph 边方向一致。
10. 未知 proposed type 收到 `contradicted` 或 `insufficient_evidence` 时保持对应不可发布状态和原始原因。
11. JSON 审计原样保留 `[]`、`false`、`0`。
12. 注入第二个 candidate/event/assertion 写入失败，验证 Chunk 级数据库事务回滚，不留下半条审计链。
13. 如果有 disposable PostgreSQL，真实执行 `SCHEMA_SQL` 两次，并验证 JSONB、FK、`ANY(%s)`、零置信度和迁移；若环境没有服务，保留集成测试并明确报告未运行原因。

不要只检查 SQL 字符串，也不要让 HugeGraph fake 永远返回空关系。测试应至少构造一个可发布和一个不可发布关系，验证最终添加的边集合。

## 验收标准

- 任意非白名单关系在 HugeGraph、关键词搜索、向量搜索、默认关系列表和首页关系统计中都不可见。
- 旧弱 `rules_verified` 不会因升级自动获得新策略信任。
- 数据库保存的 evidence 坐标能直接切出完整 Chunk 中的证据文本。
- 未验证 alias 无法参与关系发布。
- 方向纠正后不存在反向旧 description/向量文本。
- 数据库异常不会留下“candidate 已通过但事件/断言缺失”的半完成链。
- 现有严格 verifier 单元测试和完整测试不回归。
- 实际执行项目测试、编译检查、`git diff --check`；如仓库仍无 linter/type checker，如实说明。

## 最终报告

用中文给出：修改摘要、迁移策略、所有默认消费端的过滤位置、实际测试命令及数量、未运行的真实服务验证、需要运维执行的图重同步命令、风险、`git status --short` 和 diff 统计。不要 commit 或 push。
