# Knowledge Forge P1–P3 总任务：性能、抽取质量、实体归一与可验证上线

## 0. 任务定位

请在 `zjx2024/Knowledge-Forge` 最新远程 `main` 上连续完成本任务。它不是只做 P1.0，而是覆盖最初规划的全部后续目标，并加入之后确认的复杂度与运行成本优化：

- **性能基础**：可观测性、多文档图重建合并、批处理、索引、Prompt/token 和有界队列；
- **原 P1**：实体 span/ID 固定后的二阶段关系分类、显式 `NO_RELATION`、候选规划、覆盖率驱动 gleaning、多句窗口和 predicate 约束；
- **原 P2**：span-first NER、结构节点过滤、类型断言与 canonical identity 解耦、跨类型召回、cannot-link、可逆 merge/split 和 hard negatives；
- **原 P3**：金标准、实体/关系指标、证据覆盖、false-merge、Brier/ECE 校准、shadow/canary 和上线回滚。

当前已知远程基线为：

- `main@3116c0c58d146330c03572fd06d3fef590c82e31`
- tree `88f6608d0ee182624eabc0c3d94d02070ade9166`
- P0.2 自动化基线：`126 passed`

开始时必须以实际 `origin/main` 为准。如果远程已有新提交，先审查并基于最新远程继续，不能覆盖。

## 1. 完成定义

Codex 不能只给建议、只写计划或完成一个子阶段后停止。应按本文阶段连续执行：

```text
A. 仓库对齐、基线、可观测性和初始 gold
B. P0.2 图重建复杂度治理
C. 关系/实体行为等价性能优化
D. 原 P1：版本化二阶段关系管线
E. 原 P2：span-first 实体和可逆归一
F. 原 P3：扩大 gold、校准、shadow/canary 和策略激活
G. 真实服务端到端、迁移、回滚演练、完整验收和普通 Git 推送
```

每个阶段必须：

1. 更新执行计划；
2. 先补 characterization/失败测试；
3. 实现；
4. 运行本阶段测试和 P0～P0.2 核心回归；
5. 自审完整 diff；
6. 本地创建独立 commit；
7. 再进入下一阶段。

测试失败必须修复原因，不能删除、跳过、降低门槛或把 fail-closed 改成通过。除以下真实阻塞外，不要停下来等待用户逐阶段确认：

- 远程出现无法安全合并的并发变更；
- 工作区存在无法判断归属的重叠用户修改；
- 需要连接或破坏未获授权的真实用户数据库/图；
- 缺少人工确认 gold，导致 treatment 不能合法成为生产默认；这不阻止其余工程工作完成，也不允许无限等待。

外部服务缺失时继续完成代码、fake 和 disposable 集成测试，但最终必须准确区分“代码完成”“外部 smoke 未验证”“新策略尚未获准默认启用”。

本任务有两个不可混淆的完成状态：

```text
engineering_complete:
  A～G 中可由代码完成的实现、迁移、shadow、评估、回滚、fake/disposable 验证均完成；
  缺少独立人工 gold 或隔离外部服务时，treatment 保持关闭。

production_approved:
  engineering_complete；且人工确认 gold、冻结 holdout、隔离外部 smoke、shadow/canary
  和激活门禁全部通过，项目策略指针才允许切换。
```

Codex 必须尽可能达到 `engineering_complete`。只有证据真实存在时才能报告 `production_approved`；不能用模型自产标签替代人工确认，也不能因为等待人工标注而让任务无限运行。

## 2. 普通本地 Git 工作流，不使用 GitHub App

用户本地已登录 GitHub，可以正常提交和推送。本任务只使用普通 `git fetch/commit/push`，不使用 GitHub App、远程代提交或 force-push。

### 2.1 开始取证

```bash
git rev-parse --show-toplevel
git status --short --branch
git remote -v
git rev-parse HEAD
git rev-parse HEAD^{tree}
git diff --name-status
git ls-files --others --exclude-standard

git fetch --prune origin
git rev-parse origin/main
git rev-parse origin/main^{tree}
git rev-list --left-right --count HEAD...origin/main
```

记录本地/远程 commit、tree、ahead/behind、tracked 修改和所有未跟踪文件。

禁止：

```text
git reset --hard
git clean
git stash --include-untracked
git add .
git add -A
git push --force
git push --force-with-lease
```

### 2.2 处理此前本地与远程历史不同

此前本地 P0.2 commit 可能为 `33b8b36c92f32a8857f7c9386c11ac3a6a72bfac`，远程为 `3116c0c...`，但 tree 相同。若该本地对象存在，先保留：

```bash
git branch backup/pre-p1-local-33b8b36 33b8b36c92f32a8857f7c9386c11ac3a6a72bfac
```

若备份分支已存在，只核对，不覆盖或删除。

若 tracked 工作区干净或只有未跟踪任务文档，从远程历史创建工作分支：

```bash
WORK_BRANCH=codex/p1-p3-performance-quality
git switch -c "$WORK_BRANCH" origin/main
```

若存在不属于本任务的 tracked 用户修改，不 stash、不提交、不覆盖；使用新的独立 worktree：

```bash
WORK_BRANCH=codex/p1-p3-performance-quality
git worktree add ../knowledge-forge-p1-p3-work \
  -b "$WORK_BRANCH" origin/main
```

目录或分支已存在时选择新的唯一名称，不能删除现有内容；随后把实际名称保存到 `WORK_BRANCH`，后续所有 push 均使用该变量，不能又写死示例分支名。

### 2.3 任务文档和用户文件保护

以下文件默认保持未跟踪，不纳入任何 commit：

```text
Knowledge-Forge-Codex-P0-Task.md
Knowledge-Forge-Codex-P0.1-Review-Fixes.md
Knowledge-Forge-Codex-P0.2-Graph-State-Task.md
Knowledge-Forge-Codex-P1.0-Performance-Baseline.md
Knowledge-Forge-Codex-P1-P3-Complete-Program.md
```

每次提交显式列出文件：

```bash
git add -- README.md knowledge_forge/具体文件.py tests/具体测试.py
git diff --cached --check
git diff --cached --name-status
git diff --cached
git status --short
```

不得提交 DSN、凭据、真实正文、数据库 dump、模型输出日志、benchmark 临时 JSON 或用户无关修改。

## 3. P0～P0.2 永久不变量

以下约束贯穿所有阶段，不能为了性能或新功能破坏。

### 3.1 关系发布

- 未经严格语义和确定性证据验证的关系不得发布。
- 非法 JSON、漏答、未知/重复 ID、错误 evidence、错误端点、错误方向、超时和异常继续 fail closed。
- 产品查询、统计、ID 回取、关系向量 Top-K 和 HugeGraph 使用同一个策略版本发布谓词。
- 旧策略、孤儿、已删除和不可发布 relation vector 必须在 Top-K `LIMIT` 前过滤。
- indexing/outbox 失败不能降低已经验证的关系状态。

### 3.2 审计和事务

- `relation_candidates → relation_verification_events → relation_assertions` 保持 Chunk 级单事务和统一 `ingestion_run_id`。
- 任一 assertion 失败时三张表整体回滚。
- mention-level relation assertion 是不可变审计事实：固定原始 mention ID、span、evidence、候选/验证策略和 verifier 结果。实体 merge/split 不得修改它，也不得让旧 verifier 结果看似验证了新的 canonical 端点。
- mention→canonical assignment 必须版本化；产品关系、关系向量和 HugeGraph edge 是按当前 assignment/policy/generation 可重建的 canonical projection。
- 实体 mention、resolution decision 和派生计数的新批量持久化也必须有明确短事务；不得持有事务等待 Ollama、reranker 或 HugeGraph。
- 所有策略、shadow、merge/split、manual/cannot-link 决策均可审计。

### 3.3 图状态

保留：

```text
empty | stale | rebuilding | ready | failed
```

- 只有 `ready + 当前 graph policy set/generation` 才允许 `/graph/view`；其他状态返回 409。
- 完整 rebuild 仍需一次项目 purge、全量当前发布事实写入和最终三方关系数量核对。
- PostgreSQL 发布数、本次写入数、HugeGraph 实际数不一致时绝不能 ready。
- 除数量外，最终还必须核对各 label 数量、publishable relation ID/确定性摘要、边端点所属 project/generation/policy；等量但错端点、错方向、漏边或混入其他项目同样不得 ready。
- purge、Schema、任一文档写入、计数或状态切换失败均 fail closed，重试必须收敛。

### 3.4 数据与升级

- DDL additive、幂等、保留旧数据和审计历史。
- 新策略不能通过直接改旧行版本来“继承”历史验证。
- 任何影响发布集合的策略切换必须标记图 stale、递增 generation 并显式全量 rebuild。
- 不在真实用户 HugeGraph 上做 purge、失败注入或实验 shadow 写入。

## 4. 目标复杂度与理论边界

设：

- `D`：同项目连续入库文档数；
- `E`：单 Chunk 实体 mention 数；
- `R`：关系类型数；
- `M`：待归一 mention 数；
- `N`：项目 canonical entity 数；
- `K`：受硬上限保护的实体候选数。

目标：

```text
稳定批次图构建：O(一次完整项目图事实量)，不再累计 O(D²)
普通关系规划：O(E log E + 局部窗口候选)，Prompt/请求有 token 硬预算
实体归一：O(M × (indexed_retrieval(N) + K))，K 有硬上限
数据库/模型往返：随 batch 数增长，不再每 mention 固定多次连接和 HTTP
source_count：随受影响 entity/document 集合增长，不扫描全部历史 mention
```

必须承认：

1. 若同一句的 `E` 个实体确实可能两两有关，完整可能输出本身为 `Ω(E²)`；不能靠静默丢候选宣称严格线性。应使用窗口、流式/有界批次、紧凑上下文和 gold recall 门禁。
2. 一次完整 HugeGraph 重建至少为 `Ω(G)`；能消除的是重复重建和冗余网络调用，不是图事实总量。

## 5. 阶段 A：基线、可观测性与提前建立 P3-lite gold

本阶段不改变抽取、合并或发布结果。

### 5.1 修改前基线

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile knowledge_forge/*.py server.py
git diff --check
```

若配置了 disposable PostgreSQL：

```bash
KNOWLEDGE_FORGE_TEST_POSTGRES_DSN=<临时DSN> \
.venv/bin/python -m pytest -q tests/test_p02_graph_state.py -vv
```

不打印 DSN，不使用用户数据库替代。

### 5.2 结构化指标

按 Chunk/文档/项目记录且可聚合：

```text
input_tokens, chunk_count, entity_count, relation_count
sentence_count, relation_pair_checks, relation_candidates
candidate_payload_bytes, estimated_prompt_tokens, overflow_count
LLM logical calls, actual HTTP calls, repairs/retries
Embedding HTTP calls/input count, reranker calls/pair count
DB connection/statement count and classified duration
graph rebuild request/coalesced/purge/write/count calls
primary/gleaning/verifier/entity-resolution/graph duration
```

指标失败不得影响业务；不记录完整正文、Prompt、凭据或可恢复敏感数据。LLM 业务调用与 JSON repair HTTP 重试必须分开计数。

计数器必须在 PostgreSQL connection/execute、HTTP transport、reranker forward 和 HugeGraph client 等边界统一埋点，并在 fake transport/connection 测试中与其自身真实调用记录逐项核对。所有未分类调用、JSON repair、retry、outbox 和恢复重试都必须进入分母；不能只在业务函数手工 `counter += 1` 后用同一个 counter 自证性能。

### 5.3 Characterization 和初始 gold

- 固定 fake LLM/Embedding/reranker/DB/HugeGraph 响应，锁定 P0.2 的实体映射、候选顺序、事件、状态、发布集合和图计数。
- 建立 `E=0/1/2/20/50/100`、`D=1/3/10` 性能场景。
- 建立初始 50～100 个可人工审阅 Chunk，覆盖中文、英文、多词 ASCII、表格、引用、缩写、型号、版本号、同名异物、否定、假设、共现和跨句关系。
- 标注格式至少包含 mention span/type、canonical cluster、must-link/cannot-link、relation triple/direction/evidence 和无关系 hard negatives。
- 模型输出不能自动成为 gold。Codex 可以生成候选标注包，但必须将“合成/独立审核/人工确认”来源分开。
- 划分 dev 和冻结 holdout；不得在 holdout 上反复调参。

建议提交：

```text
test(perf): lock p02 construction and quality baselines
```

## 6. 阶段 B：P0.2 图重建和派生索引复杂度治理

### 6.1 持久化 graph generation 与重建合并

- 项目 PostgreSQL 图事实变化时在事务内递增 `graph_generation` 并立即置 `stale`。
- rebuild job 保存 `target_generation`；同项目只有一个有效 queued/running rebuild job。
- 同项目已经排队的一批 ingestion 在 quiescent 后只执行一次最终 rebuild。
- rebuild 开始时固定目标代次；只有当前 generation 仍等于目标、策略集合一致且三方计数一致时才能 ready。
- rebuild 期间有新 ingestion 时，旧任务不能 ready；保留/恢复 stale，并合并调度新代次。
- 服务重启后能发现 stale/rebuilding 且无有效 job 的项目并恢复。
- shadow 计算不改变正式事实，不递增 graph generation。

### 6.2 全量 rebuild 专用路径

一次 rebuild 内：

```text
ensure_schema == 1
project purge == 1
per-document purge == 0
final count check == 1
```

项目 purge 后不再每文档重复 purge/ensure_schema；空图全量重建后不做无意义全图 prune。若现有 HugeGraph 客户端有已验证批量 API，可进行有界批量写入；否则只消除确认冗余，不臆造 API。

### 6.3 Outbox 与关系向量

- relation assertion 仍是事实源，向量是派生索引。
- 审计事务成功后，publishable relation embedding 使用有界 batch；失败逐项记录 outbox，不改 verification status。
- outbox 增加分页/单次上限、`next_attempt_at`、指数退避和最大并发；服务恢复后幂等重试。
- 关系 ANN 保持策略过滤在 Top-K 前，并在 pgvector 0.8.5 启用/验证合适的 iterative scan。
- completed/永久失败记录有明确保留策略，不能无限增长。

### 6.4 测试

- 同项目预排队 1/3/10 个 ingestion，稳定代次各只进行一次最终 rebuild。
- 不同项目不错误合并。
- 等待、rebuilding、failed 均 409。
- rebuild 前/中/后新增文档、进程重启、重复 worker、purge/write/count 失败和重试收敛。
- 过期 generation 永不 ready。
- 真实/替身三方计数一致。

建议提交：

```text
perf(graph): coalesce rebuilds by persistent generation
```

## 7. 阶段 C：关系和实体行为等价性能优化

本阶段固定模型响应下必须与 P0.2 结果完全一致，可以默认启用，不升级策略版本。

### 7.1 关系候选等价优化

- 先执行廉价条件；仅在 `实体>=2、当前关系为空、正文有 cue` 且旧逻辑可能 glean 时生成候选。
- 零实体高信息补抽取不构造实体对。
- 一次建立 sentence interval，将实体按证据位置放入 bucket，只枚举旧逻辑允许的同句/相邻句。
- 保留原始 source 外层、target 内层顺序；候选字段、方向、ID、顺序、Prompt 必须逐字节等价。
- token 估算改为 streaming count，结果不变，不构造全量 CJK match 列表。
- 不在本阶段使用 `candidates[:N]`、邻居截断或新规则。

### 7.2 实体持久化和 I/O

- 一个 Chunk 的 mention、resolution decision 和受影响 `source_count` 在一个短事务批量写入；模型调用在事务外。
- `_retire_orphan_entities()` 限制到明确 project/受影响 entity IDs，不扫描全部项目。
- 删除、重入库、软删除、重试同时刷新新旧 canonical IDs。
- exact candidate 按 Chunk 批量预取；相同 normalized mention 去重但保留每条原始 mention/provenance。
- 未命中 mention 的 query embedding 使用有界批请求。
- reranker 支持跨 mention 的有界 pair batch，保持映射顺序。
- 决策仍按稳定顺序处理，session cache 和本 Chunk 新 canonical 对后续 mention 立即可见。
- 若 batch reranker 数值变化会跨阈值，该优化移入实体 shadow 阶段，不得假装等价。

### 7.3 HNSW ensure 与连接

- `_ensure_hnsw_index` 只在实际 database/schema/object_type/model/dimension 首次成功时确认并缓存；失败可重试，缓存线程安全且有生命周期。
- 新建 entity-only 等更选择性的 HNSW 索引前必须用真实计划证明；不删除兼容索引。
- 维度超过上限时暴露 degraded/exact-scan 指标和健康告警，不能伪称亚线性。
- 不持有数据库事务等待外部模型；连接/statement 数按 batch 统计。

### 7.4 `source_count`

- 不再每 mention 执行历史 `COUNT(DISTINCT document_id)`。
- 使用受影响 ID 批量派生，或建立 additive `canonical_entity_documents` 唯一来源表。
- 同一文档多个 Chunk 只贡献一次；第二文档加一；删除/重入库/重试正确；最后来源删除后实体/向量退休。
- mention/decision/count 任一步失败时短事务整体回滚。

### 7.5 关系审计写放大

- 保持 candidate/event/assertion 单事务，但将逐行 SQL 改为等价有界批量写入。
- 增加 additive `relation_verification_batches` 或等价批次记录：完整 request/response 每个 verifier batch 只保存一次，candidate event 通过 `batch_id` 引用并保存自己的 verdict/reason/error。
- 旧事件和旧 JSONB 列继续可读；不删除或伪造历史。审核查询必须能统一读取新旧格式。
- batch 失败时 candidate、batch、events、assertions 仍整体回滚。
- 不得以删除原始 verifier request/response 的审计能力换取空间。

建议提交：

```text
perf(extraction): avoid equivalent relation candidate work
perf(entity): batch resolution persistence and model io
perf(db): scope source counts and vector index checks
```

## 8. 阶段 D：原 P1——版本化二阶段关系管线

本阶段会改变候选或 Prompt，必须先 shadow，不能直接复用 `relation-verifier-p0-v1` 作为新语义标签。

### 8.1 独立版本轴和项目级启用

至少区分：

```text
entity_extraction_policy_version
entity_resolution_policy_version
relation_candidate_policy_version
relation_classifier_policy_version
verification_policy_version
schema_policy_version
```

- 项目记录上述完整 active policy set；`graph_policy_hash` 必须由六个版本、发布谓词版本和必要 schema/config 摘要确定性计算，不能只记录 verifier 或 relation classifier 版本。现有项目默认继续使用已验收 P0 策略。
- treatment 结果写专用 shadow 表/实验记录，不进入产品查询、向量、outbox 或 HugeGraph。
- shadow 失败不能影响 control，也不能标图 stale。
- 通过门禁后，先为项目全部 `indexed` 文档构建完整、版本化、隔离的 treatment snapshot；验证 snapshot 完整性后原子切换项目 policy-set 指针，再递增 graph generation、置 stale 并全量 rebuild。禁止只让未来文档使用新策略，也禁止在同一 active projection 中混合不兼容的新旧 assertion。
- 旧行保留写入时的原版本和审计标签；切换通过新 snapshot/projection 完成，不能批量改旧行版本。
- 支持回滚到 control 并再次重建。

### 8.2 Stage 1：固定实体 mention

使用当前已经验证的正文 surface/span 生成稳定 mention ID。关系阶段只引用固定 ID，不允许关系模型创造不存在的端点。

### 8.3 Stage 2：候选与显式 `NO_RELATION`

- 先枚举并计数局部 `raw pair`，再用确定性窗口、结构、schema domain/range、cue 和实体 span blocking 形成 `admitted candidate`。本文后续“候选完整性”和显式判定只指进入分类器的 admitted candidates，不要求把所有理论 `E²` 实体对都交给 LLM。
- 对每个 admitted candidate，分类器必须返回允许 predicate 或显式 `NO_RELATION`，并返回方向、精确 evidence 和 reason；漏答、重复答或未知 ID 整批 fail closed。
- 支持 directed、inverse、symmetric 和 domain/range 约束；约束集中定义并版本化。
- candidate ID 确定、可重试、跨 batch 稳定。
- blocking 参数和上限必须在 gold 上校准，并审计 `raw_pairs / blocked_pairs / admitted_candidates / classified / deferred`；blocked gold positive 计 candidate false negative。
- P1 classifier 本身必须执行严格语义判断，不能只是粗排后凭共现直接发布。复用/扩展 P0 verifier 的“谓词含义、方向、精确证据、完整 candidate decision”协议，而不是绕过它。
- classifier 输出后仍执行 deterministic evidence、端点、predicate、方向和策略验证；任何非法项 fail closed。
- 若实现选择“classifier + 独立 semantic verifier”两个模型阶段，二者调用和 token 必须有界并在基准中单独报告；只有 gold 证明额外阶段带来必要质量收益时才可默认使用。

### 8.4 Prompt 和候选复杂度

- context、entity、schema compatibility 不得在每个 pair 中重复全文；使用 `context_id/entity_id` 紧凑引用。
- Prompt 预算计算：

```text
context window
- system/schema/fixed metadata
- output reserve
- safety margin
= candidate budget
```

- 同一 context 正文在一个请求中只出现一次。
- 超预算按句子/结构边界分层拆窗，再按完整 context/window 和 admitted candidate 分批；禁止静默截断后部候选。拆窗之间保留必要 overlap，并对跨窗漏失做 gold 计数。
- 所有 batch、候选和请求有配置上限；overflow/deferred 可审计且不可发布。
- 超过系统安全上限的 Chunk 明确进入 `incomplete/deferred`，保存 raw/blocked/admitted 数量、原因、剩余范围和下一次尝试；不能把它标成抽取完成。
- `overflow/deferred` 不是成功终态：全部输入 Chunk 最终必须进入 completed/failed 等明确终态；有界重试后队列应收敛，不能通过长期积压 dense Chunk 获得更低 p95。
- gold positive 落在 overflow、deferred、timeout 或失败 Chunk 中必须计为 false negative；性能报告包含后续重试和 deferred 工作，treatment 的 deferred/overflow/error 比例不得高于 control，除非逐例证明不影响质量并有明确运维接受。
- 预算通过 gold 上“满足候选 Recall 的最小值”校准，不拍脑袋固定 200/400。
- 为输出显式预留 token，并给模型请求设置与 batch schema 相匹配的生成上限；输入校验不能占满整个 context 后让输出被截断。
- 抽取/关系数组需要确定性去重和可审计 overflow 行为，不能让模型无界输出造成数百个 verifier batch。
- 盘点 `EXTRACTION_RULE_ACCEPT_CONFIDENCE`、`EXTRACTION_MIN_RELATION_CONFIDENCE` 等现有配置：未使用的不能继续在 README 冒充有效调优项；若保留，只能赋予明确的排序/送审语义，不能凭 confidence 绕过 verifier。

### 8.5 覆盖率驱动 gleaning

- 不再只在“零关系”时判断漏抽取；根据有关系 cue 的窗口、实体覆盖和 candidate coverage 决定是否执行最多一次有界补抽取。
- 已覆盖的候选不重复请求；只处理未覆盖窗口。
- 每 Chunk 模型调用和 token 有硬预算；不能无界递归 gleaning。

### 8.6 Shadow A/B 门禁

- 同一 primary entity extraction 结果分叉给 control/treatment，避免抽取噪声。
- control 继续正式发布；treatment 只记录 exact tuple：

```text
subject canonical ID, predicate, object canonical ID,
direction, evidence span, decision/reason, tokens/calls
```

- 人工/独立复核所有 discordant cases。
- treatment 在 gold 上通过关系候选 recall、最终 P/R/F1、硬负例和成本门禁后才能按项目 canary。

建议提交：

```text
feat(extraction): add bounded versioned relation classification pipeline
```

## 9. 阶段 E：原 P2——span-first 实体与可逆归一

### 9.1 Span-first NER

- 实体模型首先输出真实正文 mention surface、精确 start/end、mention-level type assertion 和 evidence；不得先生成 canonical name 再找字符串。
- mention ID 由 document/chunk/span/surface 的稳定信息生成，重试一致。
- canonical identity 与 mention type 分离；类型错误不能直接创造另一个现实实体。
- 新策略中的 canonical identity key 不得继续把单次 mention type 当作不可变身份组成部分。现有 key/行保持兼容，通过 shadow assignment 和显式迁移处理，不能批量静默重写历史实体。
- 增加 mention/entity type assertion 的可审计表示，保存类型、证据、策略、模型/规则和 confidence；canonical 主类型是可重算的派生视图或明确决策，不覆盖冲突历史。
- 同一现实实体出现不同类型断言时进入类型裁决/灰区，而不是自动创建重复 canonical 或自动把全部历史类型改掉。
- alias 必须有独立正文 span 和可信 identity 关系；未验证 alias 不进入发布 surface。

### 9.2 结构节点过滤

- 对目录、页眉页脚、索引、参考文献、纯编号、表头/空单元格等建立可审计结构类型。
- 结构节点不能默认成为现实实体，但表格行中的事实实体和引用作者等不能被粗暴整块删除。
- 过滤规则通过结构 fixture 和 gold 验证，不按标题黑名单简单吞数据。

### 9.3 Indexed canonical name/alias

新增 additive 正规化名称表，命名可按项目风格，例如：

```text
canonical_entity_aliases(
  project_id, canonical_entity_id, entity_type,
  alias, normalized_alias, alias_kind, active,
  evidence/provenance, policy_version, timestamps
)
```

- 幂等回填 canonical name 和已有可信 aliases；旧数组继续兼容/双写。
- exact 使用 B-tree；fuzzy 使用可被 pg_trgm 利用的索引路径，禁止对项目全部实体 `unnest(array)`。
- Top-K 在数据库层生效，不能全量载入 Python。
- ambiguous exact alias 不再盲选第一条，进入有界灰区裁决。

### 9.4 跨类型、cannot-link 和 hard negatives

- 保留同类型/Generic 主 blocking。
- 增加严格有界的跨类型 exact canonical/alias 旁路，只进入 LLM/规则复核，不能直接自动合并。
- identifier、型号、版本、人物/组织等冲突形成 hard cannot-link。
- cannot-link 持久化、项目隔离、可审计，并在后续候选阶段优先阻断。
- 不进行跨类型全库 fuzzy/ANN 扫描。

### 9.5 可逆 merge/split

- canonical merge 不能只破坏性搬迁并丢失来源；增加版本化 assignment/merge event 或等价审计模型。
- 保存 from/to entity、涉及 mentions、策略、模型、scores、reason、证据、操作者、时间和可逆状态。
- 数据分为三层：不可变 mention-level relation assertion；版本化 mention→canonical assignment；可重建 canonical relation projection/vector/HugeGraph edge。merge/split 只能新增 assignment/event 并重算派生层，绝不能改写历史 assertion、candidate 或 verification event。
- 支持 split/undo，把错误合并的 mentions 重新分配，并重建受影响 canonical aliases、source_count、向量、canonical relation projection、outbox、关系端点和图。
- assignment 或 canonical 类型变化后，重新执行当前 schema 的 domain/range 检查；不再成立的派生边必须隔离或按新端点重新语义验证，不能沿用旧 canonical projection 的发布资格。
- 任一 merge/split 都递增 graph generation、置 stale，并在派生索引/图核对完成前拒绝 ready。
- merge/split 事务失败必须整体回滚；图保持 stale，直到全量重建和数量核对完成。
- 提供受项目范围约束的 service/DB API、审计查询和测试；至少能列出决策、执行 manual merge、添加 cannot-link、执行 split/undo，并触发正确的向量/outbox/graph generation。可不增加复杂人工 UI，但不能只留下无法调用的内部函数。

### 9.6 实体策略 shadow/canary 与下游重评

- `entity_extraction_policy_version` 与 `entity_resolution_policy_version` 独立、项目级启用，并纳入完整 `graph_policy_hash`。
- 实体 shadow 不得创建正式 canonical、添加 alias、移动 mention、写正式向量或污染后续候选。
- treatment 必须拥有独立、可清理的有状态 shadow namespace，保存自己的 canonical、跨文档 assignments、aliases、cannot-links、vectors、merge/split events 和下游关系端点；按真实文档顺序重放才能评估聚类传播。
- 测试必须证明 shadow 对正式表零写入、同一快照可确定性重建、失败后可清理/重跑，并能计算完整 B³、重复 canonical 率和下游 relation endpoint 差异。
- 真正 canary 必须按整个新项目/隔离 namespace 分配，不能按 mention 随机，因为早期 merge 会影响后续状态。
- treatment 通过 candidate Recall@K、pairwise/B³、false-merge、重复实体率和成本门禁后才可启用。
- span-first 会改变 mention span/ID、candidate blocking 和 canonical endpoint，因此阶段 E 完成后必须在 E 的版本化 treatment snapshot 上重新执行阶段 D 的 relation candidate gold、完整 relation shadow、discordant review 和成本门禁。此前基于旧 mention 的 D 结果只能作为历史对照，不能授权新的组合策略上线。
- 只有“新 entity extraction + 新 resolution + 对应 relation candidate/classifier/verifier/schema”完整组合通过门禁，才可按 8.1 的 snapshot/原子指针切换流程激活。

建议提交：

```text
feat(entity): add span-first extraction and versioned reversible resolution
```

## 10. 阶段 F：原 P3——完整 gold、校准和策略上线门禁

### 10.1 Gold 规模和独立性

- 从阶段 A 的 50～100 个代表 Chunk 扩展到 200～300 个；至少两个领域。
- 建议达到：`>=500` mentions、`>=150` must-link、`>=150` cannot-link、`>=200` 正关系、`>=300` 关系 hard negatives。
- 继续覆盖中英文、OCR 噪声、缩写、同名异义、型号、版本、表格、引用、否定、假设、比较、跨句和 dense Chunk。
- dev 用于调参，holdout 冻结；模型生成标签不得冒充人工 gold。
- Codex 必须提供 annotation bundle、差异报告和评估脚本；缺少人工确认时 treatment 保持关闭，不能自行宣布生产质量完成。
- 增加机器可校验的 `gold manifest`，至少记录 dataset/version、每个语料内容 hash、标签来源、reviewer 标识、`reviewed_at`、dev/holdout 划分、冻结时间和 schema/policy 兼容范围。评估脚本必须拒绝 hash 漂移、来源缺失、dev/holdout 重叠或把 synthetic/model-only 标签当作 production gate。
- 激活路径必须验证 manifest；缺少有效人工确认 manifest 时返回明确 gate failure，状态最多为 `engineering_complete`，不得由 Codex 自动补标签后绕过。

### 10.2 指标

实体抽取：

```text
exact-span micro P/R/F1
overlap F1
type macro-F1
```

实体归一：

```text
candidate Recall@K
pairwise P/R/F1
B³ P/R/F1
false-merge / missed-merge / duplicate canonical rate
```

关系：

```text
candidate Recall@budget
strict (subject,predicate,object,direction,evidence) P/R/F1
evidence coverage
invalid/legacy/failed publication count
```

校准：

```text
Brier score
ECE/reliability bins
coverage vs confidence
```

性能必须同时报告输入分母、实体/关系 gold positives、发布数、Prompt tokens、模型/DB/图调用和 p50/p95；禁止通过少处理、少抽取、少发布或长期 stale 伪造提升。

### 10.3 建议质量门禁

- invalid evidence、旧策略、verification failed 发布：`0`；
- cannot-link/hard identifier 误合并：`0`；
- critical gold relation 丢失：`0`；
- 有足够 gold 时：

```text
entity mention F1 非劣于 control 0.5 pp
B³ F1 非劣于 control 0.5 pp
relation recall/F1 非劣于 control 1.0 pp
relation candidate recall >= 99%
evidence 精确可定位 = 100%
```

如果 control 未达到绝对目标，treatment 至少不得退化，并明确后续缺口。样本不足时使用逐例零新增关键错误，不能报告虚假的小数精度。

新策略除了通过安全非劣门槛，还必须达到至少一个预先声明、与其目标对应的实质收益，例如：关系召回提高、missed merge/重复 canonical 下降、dense Prompt/token 或单位 gold positive 总成本显著下降。若质量没有改善且总成本更高，treatment 保持 shadow，不能仅凭“没有明显变差”默认上线。使用 paired discordant counts 与 bootstrap/Wilson 区间；样本小到百分点小于一两个样本时，改用零新增关键错误和逐例审查。

### 10.4 策略激活

顺序：

```text
offline gold
→ shadow
→ 隔离/新项目 canary
→ 少量项目
→ 扩大比例
→ 新项目默认
→ 旧项目显式迁移
```

每次切换记录项目策略版本，递增 graph generation，按新发布谓词全量重建，通过三方计数后才 ready。保留恢复 control 的路径和回滚演练。

激活对象是完整 policy-set snapshot，不是某一个 classifier 开关。项目 active 指针必须原子地从 control snapshot 切到已验证 treatment snapshot；任何版本缺失、文档未重处理、manifest 无效或 projection 不完整都拒绝切换。

建议提交：

```text
test(quality): add calibrated entity and relation release gates
```

## 11. 阶段 G：真实数据库、模型、图和端到端验收

### 11.1 测试层级

| 层级 | 环境 | 门禁 |
| --- | --- | --- |
| A | fake LLM/Embedding/reranker/DB/HugeGraph | 每阶段必须 |
| B | disposable PostgreSQL 16.14 + pgvector 0.8.5 | 最终必须 |
| C | fake HugeGraph + 进程重启/失败注入 | 最终必须 |
| D | 明确隔离 HugeGraph/Hubble | 环境具备时 release gate |
| E | 真实 Ollama/Embedding/reranker | 环境具备时 release gate |
| F | 人工确认 gold + shadow/canary | 宣称新策略可生产启用前必须 |

### 11.2 PostgreSQL `EXPLAIN ANALYZE`

在临时 schema/database 生成 10k 和 100k canonical entities、多个 aliases、默认维度向量并 `ANALYZE`。1M 规模只在 `RUN_LARGE_BENCHMARKS=1` 和资源充足时运行，不作为普通 CI。

执行：

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ...
```

覆盖：

- canonical name exact；
- alias exact；
- name/alias trigram Top-K；
- entity HNSW Top-K；
- relation 当前策略过滤后的 HNSW Top-K；
- source_count/文档来源刷新；
- candidate/event/assertion 事务；
- mention/decision/count 短事务。

大表选择性查询不得全表 Seq Scan；alias 不得全项目 unnest；默认支持维度必须使用 HNSW；计划门禁检查索引路径、扫描行数、buffers 和候选上限，不硬编码完整 plan 或跨硬件毫秒。

### 11.3 模型 smoke

- 记录模型名、digest、context window 和配置，不记录正文/完整 Prompt。
- structured output、candidate 完整性、evidence span、错误重试和 fail-closed 必须真实验证。
- Embedding batch 数量/顺序/维度/finite、pgvector 写入和 HNSW 召回必须验证。
- reranker pair/score 对齐、finite 和失败回退必须验证。
- 真实模型输出不要求字节级一致；质量由固定模型版本的 gold 评估。

### 11.4 隔离 HugeGraph

只使用 disposable 实例或明确独立 graph namespace。写少量完整事实，验证 label/properties/direction/ID/数量、409、purge/rebuild/retry，最后只清理测试 namespace。没有隔离实例时不得触碰用户图。

### 11.5 端到端场景

- 连续上传 1/3/10 篇文档；
- 中英文普通段落、表格、dense、跨句；
- 重复实体、别名、类型误判、型号冲突；
- 删除、重入库、任务取消、服务重启；
- graph rebuild 失败/重试；
- entity merge、split/undo、向量重建和图端点更新；
- control/shadow/canary/回滚；
- 搜索、列表、统计、关系 Top-K 和 graph view 一致。

## 12. 推荐新增测试与工具

按仓库风格调整名称，但覆盖不得减少：

```text
tests/test_p1_observability.py
tests/test_p1_graph_generation.py
tests/test_p1_relation_candidate_equivalence.py
tests/test_p1_relation_pipeline.py
tests/test_p1_entity_batching.py
tests/test_p1_postgres_plans.py
tests/test_p2_span_entities.py
tests/test_p2_entity_resolution.py
tests/test_p2_merge_split.py
tests/test_p3_quality_metrics.py
tests/test_p3_policy_rollout.py
tests/fixtures/kg_gold/
benchmarks/benchmark_kg_construction.py
benchmarks/evaluate_kg_gold.py
```

生成的报告、数据库数据和模型日志不提交。

## 13. 每阶段和最终验证命令

每阶段先运行新增 targeted tests，再运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_relation_verification.py \
  tests/test_relation_persistence.py \
  tests/test_schema_evolution.py \
  tests/test_entity_resolution.py \
  tests/test_p01_review_fixes.py \
  tests/test_p02_graph_state.py \
  tests/test_ingestion_subtasks.py \
  tests/test_document_deletion.py
```

最终使用真实 disposable PostgreSQL：

```bash
KNOWLEDGE_FORGE_TEST_POSTGRES_DSN=<临时DSN> \
.venv/bin/python -m pytest -q

.venv/bin/python -m py_compile knowledge_forge/*.py server.py

git diff --check
git diff --cached --check
```

若仓库未配置 formatter/linter/type checker，如实说明，不臆造 Ruff/Mypy。若新增这些工具，必须有配置、固定依赖并对全仓运行，不能只检查新文件。

Live 测试使用显式环境开关，例如：

```bash
RUN_LIVE_MODEL_TESTS=1 .venv/bin/python -m pytest -q -m live_model
RUN_HUGEGRAPH_INTEGRATION=1 .venv/bin/python -m pytest -q -m hugegraph_integration
RUN_LARGE_BENCHMARKS=1 .venv/bin/python benchmarks/benchmark_kg_construction.py
```

实际 marker/命令按实现为准，并在 README 与最终报告给出。

## 14. 性能硬门禁

CI 以操作数、增长率、payload/token 和查询计划为硬门禁；绝对耗时只在同硬件记录 before/after。

- 无需 gleaning：pair checks、candidate payload、gleaning calls 均为 0。
- compact relation Prompt 中相同 context 只出现一次；请求不超过 context budget，输出预留和 safety margin 非零。
- overflow 分窗/分批或明确 deferred，不能静默丢弃。
- exact entity hit 不调用 Embedding/reranker/LLM。
- Embedding HTTP 随 batch 数增长；reranker pairs `<= M × configured_cap`。
- DB connection 随 Chunk/batch 增长，不再每 mention 新建多次连接。
- source_count 不随历史 mention 总量重扫。
- 同项目稳定 `D=10` 批次从 10 次完整 rebuild 降为 1 次；并发新代次允许额外一次，但旧代次不能 ready。
- cache、queue、batch、shadow 和 outbox 均有上限、TTL/清理或持久化生命周期。

同硬件 release 报告建议目标：普通语料 p95 不恶化超过 5%，dense 语料 p95 改善至少 20%，候选 Prompt tokens 下降至少 40%，实体阶段 DB/Embedding 往返下降至少 50%。未达到时继续分析迭代，不能只报告总时间。

## 15. 明确禁止

- 禁止用 `candidates[:N]`、只取前 N 个实体或静默丢弃后部窗口换速度。
- 禁止放宽 verifier、发布旧策略、后移 Top-K 策略过滤或取消图数量核对。
- 禁止全项目实体/向量加载到 Python 两两比较。
- 禁止未经 gold/shadow 修改候选、Prompt、阈值、Top-K、模型、Chunk 大小或类型规则后直接默认启用。
- 禁止实体 shadow 污染正式 canonical/alias/mention/vector。
- 禁止关系 shadow 进入正式 assertion/search/graph。
- 禁止无界并发 Ollama、reranker、PostgreSQL 或 HugeGraph。
- 禁止把自己生成的模型标签冒充独立人工 gold。
- 禁止在用户真实数据库/图上生成规模数据、purge 或失败注入。
- 禁止删除失败测试、改成无意义断言或将 skipped 伪报为 passed。
- 禁止 force-push、覆盖远程新提交或提交任务文档/凭据/临时结果。

## 16. 阶段提交与普通 Git 推送

建议阶段 commits：

```text
test(perf): lock p02 construction and quality baselines
perf(graph): coalesce rebuilds by persistent generation
perf(extraction): avoid equivalent relation candidate work
perf(entity): batch resolution persistence and model io
feat(extraction): add bounded versioned relation classification pipeline
feat(entity): add span-first extraction and reversible resolution
test(quality): add calibrated entity and relation release gates
docs: document p1-p3 policies benchmarks and rollback
```

每阶段 targeted/core tests 通过后创建本地 commit，并用普通 Git 将工作分支作为远程检查点：

```bash
git push -u origin "$WORK_BRANCH"
```

后续：

```bash
git push origin "$WORK_BRANCH"
```

不得因为阶段分支已推送就把未通过的 treatment 设为默认。

全部阶段、真实 PostgreSQL、完整回归、编译和自审完成后：

```bash
git fetch origin
git rev-list --left-right --count HEAD...origin/main
git merge-base --is-ancestor origin/main HEAD
```

只有祖先检查成功才执行普通非 force 推送：

```bash
git push origin HEAD:main
```

若远程变化或 non-fast-forward：

1. 不使用 force；
2. `git fetch origin`；
3. 审查 `git log --left-right HEAD...origin/main` 和 diff；
4. `git merge --no-edit origin/main`，保留已经推送的阶段 commit 历史；
5. 安全解决冲突；语义不明则报告阻塞；
6. 重新运行受影响测试、完整 pytest、编译和 diff check；
7. `git push origin "$WORK_BRANCH"` 更新工作分支检查点；
8. 再次确认 `origin/main` 是 HEAD 祖先，然后普通 `git push origin HEAD:main`。

禁止在已经推送阶段检查点后 rebase，因为那会要求重写远程工作分支。若开始时选择了唯一分支名，先在当前 shell 重新设置同一个 `WORK_BRANCH`；不得误推到示例名称。

推送后：

```bash
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
git rev-parse HEAD^{tree}
git rev-parse origin/main^{tree}
git ls-remote --heads origin main
git status --short --branch
git ls-files --others --exclude-standard
```

预期 `HEAD == origin/main`、tree 相同、未跟踪任务文档仍保留，且未使用 force。

## 17. README 与运维文档

必须说明：

- graph generation/coalescing 和 pending graph 状态；
- 策略版本、control/shadow/canary/active；
- relation/entity Prompt 与 batch 配置；
- alias/source-count 迁移和幂等回填；
- HNSW 支持维度、degraded 模式和 EXPLAIN；
- gold/benchmark/live smoke 命令；
- merge/split/rollback；
- 项目策略升级如何触发 stale/rebuild；
- 哪些已真实验证，哪些需要人工 gold 或隔离外部服务。

不要求破坏性重入库；历史项目采用显式、可重试、逐项目迁移。

## 18. 最终报告

用中文提交可核验报告：

1. 总结果和各阶段是否完成；
2. 原 P1、P2、P3 目标逐项对照；
3. P0～P0.2 不变量证据；
4. before/after 复杂度与操作数表；
5. entity mention、聚类、relation、evidence 和 calibration 指标；
6. shadow discordant cases、canary 和启用状态；
7. PostgreSQL EXPLAIN 摘要；
8. HugeGraph/Ollama/Embedding/reranker 真实验证范围；
9. DDL、迁移、回填、policy/generation 和回滚；
10. 每条测试命令的 passed/skipped/failed 数量；
11. 未验证项和仍存在风险；
12. 每阶段 commit SHA 和文件清单；
13. 开始/最终本地与 `origin/main` commit/tree；
14. 工作分支、普通 push 命令和结果；
15. 明确确认未 force-push；
16. 未跟踪任务文档仍保留。

如果没有人工确认 gold 或隔离外部服务，必须写清：工程、shadow、评估和回滚能力是否完成，以及哪些新策略仍保持关闭。不能把“代码已实现”表述为“生产质量已经得到独立证明”。
