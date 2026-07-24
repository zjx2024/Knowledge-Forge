# Knowledge Forge：实体关系召回、归一化与 GraphRAG 社区闭环持续执行任务书

> 本文件既是给 Codex 的主提示词，也是跨会话执行计划。不要只分析或重写本计划；读取后应在目标仓库中实际实施、测试、记录证据，并按依赖顺序持续推进。

## 一、推荐启动与续跑命令

先把本文件放到本机可访问的位置，再设置绝对路径：

```bash
export REPO="/absolute/path/to/Knowledge-Forge"
export PROMPT_MD="/absolute/path/to/codex-continuous-entity-relation-community-plan.md"
export RUN_DIR="${TMPDIR:-/tmp}/knowledge-forge-codex-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

codex --version
codex exec --help
set -o pipefail

codex exec \
  -C "$REPO" \
  --sandbox workspace-write \
  -c 'approval_policy="never"' \
  --json \
  --output-last-message "$RUN_DIR/final.md" \
  - < "$PROMPT_MD" \
  | tee "$RUN_DIR/events.jsonl"

jq -r 'select(.type == "thread.started") | .thread_id' \
  "$RUN_DIR/events.jsonl" | head -n 1 > "$RUN_DIR/session_id"
```

不要加 `--ephemeral`，否则不能可靠恢复会话。`--full-auto` 已是弃用兼容参数，不使用它；普通工作机也不要使用 `--yolo` 或 `danger-full-access`。

如果一次运行未完成，使用明确的 session ID 续跑：

```bash
export SESSION_ID="$(cat "$RUN_DIR/session_id")"
set -o pipefail

codex exec \
  -C "$REPO" \
  --sandbox workspace-write \
  -c 'approval_policy="never"' \
  --json \
  --output-last-message "$RUN_DIR/final-resume.md" \
  resume "$SESSION_ID" \
  "读取仓库中的执行计划、状态账本、git 状态和最近门禁证据；核验后从 next_action 继续实际实施。不要重做已验证阶段，不要只汇报计划。持续到所有可完成阶段完成，或只剩有证据的外部阻塞。" \
  | tee -a "$RUN_DIR/events.jsonl"
```

同一仓库没有并行 Codex 会话时，才可用 `resume --last` 代替明确的 session ID。非交互模式无法临时等待人工审批；网络、Docker socket、推送或工作区外写入被拒绝时，记录为对应外部门禁阻塞，继续所有不依赖它的工作。不要为了绕过权限而放宽沙箱。

---

## 二、给 Codex 的最高优先级执行指令

这是一个实施任务，不是方案咨询。你的任务是先解决当前实体—关系抽取数量少、归一质量不可可靠测量的问题，再完成 GraphRAG 社区检测复验、community reports 和显式 opt-in 的 Global Search shadow 工程闭环。

持续遵守以下规则：

1. 先读取全部适用的 `AGENTS.md`、仓库文档、迁移、测试、配置和当前代码，再做改动。
2. 使用计划工具，并在仓库内维护可恢复状态；上下文压缩、进程中断或新会话后必须从状态继续。
3. 每阶段先做定向测试，再做该阶段门禁。门禁失败时定位、修复并重跑，不能把失败测试改成跳过来伪造通过。
4. 不因某个外部服务不可用而停止所有工作。把 live-model、Docker、网络、远程推送等阻塞局限在对应门禁，继续可离线完成的实现和测试。
5. 不得越过阶段的 Stop/Go 条件。工程实现完成、Gold 通过、live 模型验证、生产批准是不同状态，必须分别报告。
6. 常规实现细节自行作出保守、可逆的判断；只有真实凭据、破坏性操作、产品策略切换或需求扩张才需要用户决定。
7. 每次结束前必须留下准确的 `next_action` 和一条可直接恢复的命令，不能只留笼统 TODO。

## 三、已知基线与第一轮检查

用户给出的历史基线提交为：

```text
74a12308a92d16ff93969f28e76c82fb90ab5643
```

该提交包含 GraphRAG BYOG 社区检测第一阶段和 `docs/entity-relation-quality-assessment.md`。它只是历史锚点，不得假定当前 `HEAD` 仍等于该 SHA，也不得为了回到该 SHA 而 reset、覆盖或删除用户工作。

开始时执行并记录：

```bash
pwd
git status --short --branch
git rev-parse HEAD
git rev-parse --verify 74a12308a92d16ff93969f28e76c82fb90ab5643
git log --oneline --decorate -n 20
git remote -v
git fetch --prune origin
git rev-parse origin/main
git merge-base HEAD origin/main
git diff --check
git diff --name-status
```

若 fetch 因网络或权限失败，记录准确错误并基于现有 refs 继续本地工作。若工作树已有修改或未跟踪文件：

- 视为用户资产，禁止删除、覆盖、清理或顺手格式化；
- 不使用 `git reset --hard`、破坏性 checkout、force-push；
- 只显式暂存本任务产生且已核对的文件；
- 若已有修改与本任务重叠，先读 diff 并保留用户意图，无法安全合并时才报告阻塞。

不得直接在 `main` 上提交。优先使用或恢复功能分支：

```text
feat/entity-relation-recall-community
```

如果当前已经位于承载本工作的非 main 功能分支，优先复用而不是制造重复分支。

## 四、持久计划和状态账本

如果仓库没有等价机制，创建：

```text
docs/plans/entity-relation-community-execplan.md
.codex/work/entity-relation-community-state.yaml
.codex/work/entity-relation-community-evidence/
```

计划文档记录设计、阶段和人工可读证据；YAML 是机器可读断点。`.codex/work/` 默认不提交、不暂存，除非仓库已有明确相反约定；不要为此改动项目级 `.gitignore` 而影响其他人。

状态至少包含：

```yaml
run_id: ""
base_sha: ""
start_head_sha: ""
origin_main_sha: ""
branch: ""
current_phase: P0
phase_status: in_progress
engineering_complete: false
gold_gate_passed: false
shadow_isolation_passed: false
community_detection_passed: false
community_reports_live_verified: false
global_search_release_ready: false
production_approved: false
graph_generation: null
policy_hash: ""
input_hash: ""
gold_hash: ""
migration_head: ""
completed_commits: []
commands_run: []
blocked_checks: []
modified_files: []
next_action: ""
next_command: ""
```

每个阶段开始和结束都更新：当前 SHA、分支、工作树、输入与配置 hash、执行命令、退出码、实际测试数量、修改文件、数据库/容器/临时资源、阻塞项和下一动作。不得仅凭旧文字将阶段标为完成；恢复时要核对 Git、迁移和最近成功门禁。

## 五、全局数据与安全不变量

这些条件在所有阶段都成立：

- 新关系候选只来自当前文档当前 active revision 的 mention 和受限上下文窗口。
- 允许新文档中的 mention 归一到项目历史 canonical entity；只有新文档同时提及两个端点并提供关系证据时，才能形成“新实体—旧实体”或“旧实体—旧实体”的新断言。
- 禁止把新实体与数据库全部旧实体做笛卡尔积，禁止把跨文档共现当成关系证据。
- shadow treatment 不得写正式实体、关系、向量、outbox、manifest、graph generation、社区、报告或现有查询状态。
- 不得为了提高关系数量而放宽正式图 fail-closed 发布门禁。
- `NO_RELATION`、`OTHER_UNSUPPORTED`、`uncertain`、`retryable_failed` 不得发布，也不得强制降级成 `related_to`。
- LLM 超时、不可用、格式错误或解析失败不得创建正式 canonical entity，不得被记成语义拒绝。
- current control、既有 P1/P2 shadow、旧 revision、审计记录和用户数据必须保持可追溯，不得暗改或清空。
- treatment 未正式激活前，正式社区只消费 current publishable control；若 treatment 后续激活，必须重新生成 graph generation、community detection、reports 和 Global Search 验证，不得复用旧快照。
- 不能读取、打印、提交真实密钥；文档和 fixture 不包含密钥、本机绝对路径或真实敏感正文。

## 六、阶段依赖和完成定义

```text
P0 基线与漏斗
  → P1 评估器修复
  → P2 实体归一 shadow
  → P3 关系召回 shadow
  → P4 A/B 与晋级判断

P4 未晋级时：社区继续消费 current control
P4 晋级后：生成新的正式图快照

上述任一路径
  → P5 社区检测复验
  → P6 Community Reports shadow
  → P7 Global Search opt-in shadow
  → P8 全量门禁、文档、提交与交付
```

若缺少足够人工 Gold，P4 只能得到 `pilot/engineering complete`，不得把 treatment 切为 active；但 P5–P7 可以基于 current control 快照继续完成工程闭环。若将来 treatment 获准激活，P5–P7 必须针对新快照重跑。

---

## P0：冻结基线，找出关系实际丢失位置

### 工作内容

1. 完整读取：
   - `docs/entity-relation-quality-assessment.md`；
   - 实体/关系抽取、sanitizer、验证、归一、发布投影和 graph manifest 代码；
   - 当前 P1/P2 shadow、release gate、社区检测模块和数据库迁移；
   - pyproject/requirements、Docker/Compose、现有测试与脚本。
2. 区分三套统计：旧运行库原始数据、current publishable control、P1/P2/treatment shadow。旧运行的 374 条关系、`related_to` 比例或“0/291”等数字不得作为新方案准确率。
3. 为固定文档集建立可复现 manifest，至少记录：语料/revision hash、dev/holdout 划分、Gold 版本、代码 SHA、迁移 head、policy/prompt/model/config/evaluator hash。
4. 增加端到端状态漏斗，每条候选必须恰好进入一个终态：

```text
raw_entity_mentions
→ admitted_mentions / rejected_mentions / uncertain_mentions
→ entity_resolution_candidates
→ resolved / new_shadow_canonical / uncertain / retryable_failed
→ bounded_relation_pairs
→ specific_relation / NO_RELATION / OTHER_UNSUPPORTED / uncertain / retryable_failed / rejected
→ sanitized
→ endpoints_resolved
→ verified
→ publishable
```

5. 报告每层 count、唯一 ID、失败原因和守恒检查，不能只看最终关系条数。

### 退出门禁

- 当前代码和数据口径能复现；
- 旧数据不再被标成当前正式图；
- 漏斗状态互斥且计数守恒；
- baseline manifest 与 hash 已落盘；
- 定向测试通过。

---

## P1：修复评估器和确定性缺陷

### 必做修复

1. 修复 P1 类型规则中 `"*"` 通配符不能匹配的问题，并增加 BASE_SCHEMA 多类型回归测试。
2. 修复 alias、大小写和证据定位的一致性问题，但不能通过宽松模糊匹配掩盖 span 错误。
3. release evaluator 必须真正使用：
   - `must_links`；
   - `cannot_links`；
   - `hard_negatives`；
   - 实体归一 candidate Recall@K；
   - 关系 pair candidate Recall@budget/K。
4. 纠正指标命名。原来测关系端点候选的指标不能继续称为实体归一 `candidate_recall`。
5. 输出混淆矩阵、分母、TP/FP/FN、原始样本 ID 和置信区间；测试数量不写死。
6. 若现有 Gold 不足，生成待人工标注队列、格式校验器和说明；不得由模型凭空补出 Gold 并把它当人工真值。

### Gold 冻结规则

- control 与 treatment 使用完全相同的 active-revision 输入 universe；
- holdout 阈值必须在查看 treatment holdout 结果前登记；
- 关系标签应在限定上下文内尽量穷尽；否则 treatment-only 新关系必须进入人工仲裁队列，不能擅自算 TP；
- strict relation key 至少包括 revision/content hash、subject canonical、predicate、object canonical、方向和 exact evidence span；方向或证据错误不算 TP；
- 小样本只能标记 pilot，不声称统计显著。

如果仓库没有既定 production Gold 下限，可把以下作为默认建议而不是伪造完成条件：200–300 个 chunks、至少 500 mentions、150 must-links、150 cannot-links、200 条正关系、300 hard negatives、至少两个领域、双人复核及争议仲裁。数据不足时继续完成工程实现，但 `gold_gate_passed=false`。

### 退出门禁

- 所有 Gold 标签真实参与计算；
- 指标名称与实现一致；
- fixture 覆盖 must-link、cannot-link、hard-negative、别名、同名异义和 span；
- 状态守恒；
- 定向与无外部服务测试通过。

---

## P2：实体 mention、准入和归一 shadow v2

### 目标语义

把 occurrence、surface/type assertion 和 canonical 决策拆开：

```text
revision + chunk + exact mention span
→ stable occurrence
→ versioned surface/type/confidence assertion
→ entity admission
→ candidate generation
→ canonical decision or explicit non-terminal state
```

### 必做实现

1. occurrence ID 由 revision、chunk 和精确 span 稳定生成；surface、type、confidence 变化不得静默制造一个全新 occurrence。
2. 增加 entity admission，区分：
   - 合法 mention；
   - 值得建立/复用 canonical；
   - 低价值、标题、长定义或通用短语；
   - uncertain/retryable。
3. 同名同类型且项目内唯一不再无条件自动合并。使用别名、hard identifier、类型兼容、局部上下文和来源证据；同名人物/机构 hard negative 必须不误合。
4. 名称规范化保留有语义的标点和标识符，例如 `C++`、`C#`、`.NET`，不能全部折叠为同一短串。
5. 类型兼容矩阵和 alias 规则版本化；记录候选、特征、决策、policy hash 和 provenance。
6. LLM 超时、限流、无效 JSON、无法解析时进入 `retryable_failed/uncertain`；不得因此创建正式 canonical。
7. 先写独立 shadow 表或明确 shadow namespace；不得变更 control/P1/P2 的正式投影。
8. 新文档序列 fixture 必须覆盖：
   - 可靠 alias 复用旧 canonical；
   - 同名异义不合并；
   - hard-ID 冲突；
   - LLM 失败不创建正式 canonical；
   - revision 更新不会让旧 revision 继续进入 current 图。

### 隔离证明

在 disposable PostgreSQL 中尽可能给 shadow worker 一个正式表 SELECT-only、shadow 表可写的独立角色。shadow 前后对正式 canonical、alias、mention、assignment、type、relation candidate/event/assertion/projection、向量、search materialization、outbox、graph state/manifest 做排序 manifest/hash 和 row count 对比，必须完全一致。主动尝试写正式表应由数据库拒绝。

### 退出门禁

- shadow 失败注入和恢复幂等；
- 正式表、graph generation、现有 API 查询结果均不变；
- cannot-link/hard identifier fixture 零误合；
- 所有正式端点 lineage 可定位；
- `shadow_isolation_passed=true` 只能在真实 disposable DB 证明后设置。

---

## P3：关系召回 shadow v2

### 核心架构

保留现有联合抽取作为 control；新增可诊断 treatment：

```text
当前文档当前 revision 的 admitted mentions
→ 同句 / 相邻句 / 相邻 chunk 边界的有界实体对
→ 类型、距离、Top-K 与预算过滤
→ 受控谓词 | NO_RELATION | OTHER_UNSUPPORTED
→ exact evidence、方向和端点验证
→ supported / uncertain / retryable_failed / rejected
→ shadow publishable projection
```

### 候选生成

- 默认只生成当前 revision 内实体 mention 对；
- 同句最高优先，相邻句用于承接和指代，相邻 chunk 仅处理切分边界；
- 按类型兼容、文本距离、段落和每实体 Top-K 限制，避免 (O(n^2)) 无界膨胀；
- 候选预算、截断原因、deferred 状态、tokens、调用次数、p50/p95 latency 全部记录；
- 超预算必须可恢复，不能静默丢弃正例；
- 跨文档连接只能通过当前 mention 归一到旧 canonical 实现，不能枚举历史实体。

### 受控分类与修复

1. 分类输出只能为已版本化关系 schema 中的具体谓词、`NO_RELATION` 或 `OTHER_UNSUPPORTED`，并返回方向、端点、最小充分证据和置信度。
2. 未知谓词、无关系和证据不足不得转换为 `related_to`。
3. 补抽从“关系数为零才触发”改为覆盖率驱动。即使已有一条关系，只要存在未连接高质量实体、未判断候选、只有泛化谓词、句内多实体无边或大量孤立节点，也可触发定向修复。
4. 阈值必须配置化并视作实验初值，不包装成行业标准。可从以下 shadow 条件开始，经 dev Gold 校准后冻结：

```text
entity_count >= 3 and (
  relation_count == 0
  or linked_entity_ratio < 0.60
  or candidate_coverage < 0.50
  or specific_relation_ratio < 0.50
)
```

5. 修复轮只判断尚未覆盖的候选或修复具体失败原因，不重新自由生成全文关系。
6. 增加轻量指代/别名支持，但每个端点仍要有可定位 mention span，evidence 不能是一大段无法核验的文本。
7. 状态至少区分：`supported`、`no_relation`、`other_unsupported`、`uncertain`、`retryable_failed`、`rejected`；每个候选恰好一个当前终态，重试保留 attempt history。

### 必测 fixture

- 首轮已有一条关系但仍漏多条，补抽能恢复；
- 新实体与旧 canonical 同时被新文档提及并有证据，可形成新断言；
- 新文档只提到一个端点时，不与数据库旧实体凭空配对；
- 同句、邻句、相邻 chunk、指代、方向相反、无关系、unsupported、证据不精确；
- classifier/verifier 超时、非法 JSON、持久化中断与 resume；
- 重复运行幂等且无逻辑重复 assertion。

### 退出门禁

- relation candidate Recall 能独立测量；
- 补抽不再受 `relation_count == 0` 限制；
- 正式 control 无变化，shadow 隔离证明仍通过；
- 关系总数、每千字关系数和 linked ratio 只作诊断，不能单独作为通过依据；
- 定向、失败注入和 PostgreSQL 集成测试通过。

---

## P4：冻结 A/B、Gold 门禁和可选晋级

### A/B 要求

在同一 corpus/revision/Gold/policy universe 上比较 current control 与 treatment v2。至少报告：

- entity exact mention Precision/Recall/F1；
- entity candidate Recall@K；
- B³ 或等价 cluster 指标；
- must-link、cannot-link、hard-negative；
- relation candidate Recall@budget；
- strict relation Precision/Recall/F1；
- endpoint、direction、predicate 和 exact evidence 正确率；
- 具体谓词比例、`related_to` 比例；
- treatment-only 的 `true_added`、`false_added`、`missed_recovered`、`spurious_added`；
- tokens、调用数、延迟、失败率和重试成本。

### 默认 production 建议门槛

若仓库已有经批准门槛，沿用已有门槛，不擅自替换。否则先把以下门槛写入冻结 manifest，再查看 holdout：

- entity candidate Recall@K ≥ 99%；
- cannot-link/hard-identifier 误合并为 0；
- relation candidate Recall@budget ≥ 99%；
- strict relation recall 相对 control 至少提高 3 个百分点，配对单侧 95% 下界大于 0；
- strict precision 点值 ≥ 95%，Wilson 下界 ≥ 90%，且相对 control 的下降下界不超过 1 个百分点；
- strict F1 改善，且不能以 evidence、方向或端点错误换取；
- hard-negative false positive 为 0；
- exact evidence 可定位率 100%；
- invalid evidence、unknown predicate、wrong policy、verification-failed 的发布数为 0；
- publishable `related_to` 不高于 control，且 `related_to` 不计入召回收益。

如果样本不足以支持这些区间，只能保持 pilot。不得在看过 holdout 后反复修改门槛或用更多关系数量替代质量门禁。

### 晋级规则

- 门槛未通过：保留 treatment shadow，记录失败类别和下一轮实验；`gold_gate_passed=false`，不得切 active。
- 门槛通过但没有产品批准：实现为可选择、可回滚的 policy/feature flag，默认 control 不变；`production_approved=false`。
- 只有 Gold 通过、数据库隔离通过且用户/产品明确批准后，才可把 treatment 设为 active；激活必须生成新的 graph generation，并使旧社区/报告失效。

无足够人工 Gold 不阻塞后续社区工程，但后续必须读取 current control 的 publishable 图。

---

## P5：GraphRAG BYOG 社区检测复验

基于仓库锁定的 `graphrag==3.1.1` 和实际安装接口工作，不凭记忆猜 API。先探测 CLI/API 签名，使用公开稳定入口，并记录实际命令和版本。

### 正式输入语义

只读取：

- current active revision；
- canonical projection 可发布；
- manual verified，或当前 policy 版本的 rules/LLM verified；
- 与指定 project、graph generation、policy 和 input hash 一致的数据。

旧 revision、unpublishable、P1/P2/treatment shadow 输入数必须为 0。treatment 未激活时只使用 control。

### 检测规则与测试

- 权重保持并验证 `publishable_assertion_count_v1`；
- 重复端点边聚合权重，自环策略明确，孤立节点保留，空图成功产生 0 个社区；
- fixture：两个稠密小团簇、团簇间一条弱边、固定 `seed=42`，连续运行两次；
- 比较规范化成员集合和父子层级，不比较 GraphRAG 随机 UUID；
- 验证 size、成员、社区内关系、父子引用、edge provenance 完整；
- graph generation、policy 或 input hash 变化时旧 run stale/superseded；不变化时幂等复用；
- detection 过程必须证明没有调用 LLM/Ollama。

### disposable PostgreSQL

从最新 schema 迁移隔离数据库，写入 project/document/revision/entity/relation fixture，实际执行 detect，验证 run/community/membership/edge provenance、failed/superseded 生命周期，以及原实体、关系、manifest、P1/P2/treatment 状态不变。只清理本轮创建的数据库、容器和临时文件。

完成后设置 `community_detection_passed=true`，否则不得进入 active community reports。

---

## P6：Community Reports shadow 闭环

### 实现要求

1. 核对 GraphRAG 3.1.1 实际公开 API/CLI 和输入/输出 schema，再实现 `create_community_reports` 适配；不调用未验证的内部私有函数。
2. 报告必须绑定精确快照，版本键或等价唯一约束至少覆盖：

```text
project_id
graph_generation
community_detection_run_id
community_id
level
community_input_hash
prompt_hash
model_config_hash
```

3. 运行状态至少支持：

```text
pending → running → succeeded
                  ↘ retryable_failed
                  ↘ terminal_failed
                  ↘ superseded
```

4. 记录 GraphRAG 版本、模型标识、prompt/config hash、attempt、输入/输出 hash、token/cost（若提供）、时延、错误类别和完整 provenance；禁止保存明文密钥。
5. 批次幂等，可从失败 community 续跑；已成功且输入 hash 一致的报告不重复调用模型。
6. 只有所有 eligible community 都成功、schema 和引用校验通过时，报告 run 才能原子 active。partial/failed run 不得替换旧 active run。
7. 旧报告保留审计；graph generation 或 detection run 改变后不能作为当前报告。
8. 报告中的 entity/relation/community citation 必须解析到同一 project 和快照，并最终追溯到 active document revision 与 evidence。跨项目、shadow、旧 generation 引用为 0。
9. 空图产生成功的 0-report run，模型调用数为 0。
10. 先做固定 fake model 的确定性、无效 JSON、超时、限流、缺引用、持久化中断和 resume 测试；有真实模型配置时再运行 `live_model`，不得用 mock 冒充 live 通过。

### 语义质量门禁

除了 JSON schema，还要在固定人工 fixture 上核查每个 finding 的 claim-level support、citation coverage、supported claim precision 和 critical hallucination。建议：citation coverage 100%、supported claim precision ≥ 95%、critical hallucination 为 0。没有人工报告 Gold 或真实模型时可以标记 `engineering_complete=true`，但必须保持：

```yaml
community_reports_live_verified: false
production_approved: false
```

P6 完成前不得接入现有检索和问答。

---

## P7：Global Search 显式 opt-in shadow

社区“做完”包括一个可调用、可测试但默认不影响生产问答的 Global Search shadow。使用 GraphRAG 3.1.1 实际支持的 map-reduce/global search 入口，不自创不兼容协议。

### 实现要求

- 通过独立 CLI/API 或明确 `mode=global_shadow`/feature flag 调用；默认搜索与问答路径行为和结果保持不变；
- 只读取当前 project 下 active、完整且 graph generation/policy/input hash 与当前图完全一致的 report run；
- stale、failed、partial、superseded 或跨项目报告一律 fail closed，返回明确 409/503 或仓库一致的等价错误，不静默混用旧报告；
- 默认 `allow_general_knowledge=false` 或 GraphRAG 3.1.1 的等价设置；
- 响应返回 snapshot/report run、被选 report IDs、可解析 citations 和运行 provenance；
- 记录 query hash、模型/prompt/config、选中 reports、map 输出、reduce 输出、tokens、时延和失败状态；
- map/reduce 任一步失败时不能把 partial answer 标成 success；
- 覆盖 token cap、batch、ranking/filter、malformed output、timeout/retry、不可回答、跨项目和跨 generation 测试。

### 发布门禁

若没有独立人工 query Gold 和真实模型，只能设置：

```yaml
global_search_release_ready: false
production_approved: false
```

建议 production query Gold 至少包含多社区综合、不可回答、对抗和跨项目问题；实质 claim citation coverage 100%、citation factual support ≥ 95%、critical hallucination 为 0、预置不可回答题正确 abstain 100%，并与现有 baseline 做配对盲评。LLM-as-judge 只能作为诊断，不能单独批准上线。

### 必做黑盒联动测试

在同一 disposable DB 中：

1. 完成 control detection + reports；
2. 只写 treatment shadow 实体/关系，验证正式 graph generation、community input hash、active report run 和 Global Search 上下文完全不变；
3. 通过测试专用受控路径激活一条 publishable 关系，验证 graph generation/input hash 改变、旧 detection/report 立即 stale/superseded、Global Search 在新报告完成前 fail closed；
4. 新 detection + reports 全部成功后才恢复 Global Search。

该测试同时证明 shadow 不泄漏、社区不会误用旧快照、查询不会跨代混合。

---

## P8：全量门禁、文档、提交和推送

### 强制验证

先自动发现并遵循仓库实际命令；以下为最低门禁，不能用旧测试数量代替：

```bash
.venv/bin/python -m pytest -q -ra --strict-markers \
  -m "not integration and not live_model and not hugegraph_integration" \
  --durations=25

.venv/bin/python scripts/run_disposable_postgres_tests.py
.venv/bin/python -m pip check
.venv/bin/python -m py_compile knowledge_forge/*.py scripts/*.py server.py
git diff --check
```

还要执行仓库已有 Ruff、Mypy、迁移、Compose 等正式门禁。构建 CPU Docker 镜像并验证：

- 容器内 Python 3.12；
- `graphrag==3.1.1`；
- 新模块可导入；
- 原 API 健康检查通过；
- 默认检索/问答在 feature flag 关闭时无新调用、无结果变化。

本轮不强制 GPU、Ollama、reranker 或 HugeGraph live；没有真实 community-report/global-search 模型配置时也不强制付费 live 测试，但必须准确列为未验证，不能声称通过。社区 detection 的“未调用 LLM”证明只适用于 P5；P6/P7 本来就依赖模型，应分别报告 fake 与 live 结果。

### 文档

新增或更新最小必要文档，说明：

- control 与 relation/entity treatment shadow 架构；
- 新文档当前 revision 候选实体对语义，以及如何通过归一复用旧 canonical；
- 状态漏斗、`NO_RELATION`、`OTHER_UNSUPPORTED`、uncertain/retryable；
- 正式图 publishable 过滤与 fail-closed 规则；
- GraphRAG detection、community reports、Global Search 的快照依赖；
- CLI/API 用法、迁移、新表、状态生命周期、失败恢复；
- 空图、孤立点、完整快照重建与 supersede；
- 哪些功能仍是 shadow、哪些 live 验证未完成、生产默认是否改变。

### 提交

每个独立阶段只在定向门禁通过后提交。提交前执行：

```bash
git status --short --branch
git diff --check
git diff --name-status
git diff
git diff --cached --check
git diff --cached --name-status
```

只显式暂存本阶段文件，禁止 `git add .`，禁止暂存用户已有未跟踪任务文档和 `.codex/work/` 运行状态。建议提交信息按实际阶段选择：

```text
test: fix entity relation quality evaluation
feat: add shadow entity relation recall pipeline
feat: add versioned community reports
feat: add opt-in graphrag global search
```

不要为了匹配建议信息而把多个未验证阶段塞进同一提交。

### 推送

只有对应提交的所有可运行门禁通过后才考虑推送。推送前：

```bash
git fetch --prune origin
git status --short --branch
git merge-base --is-ancestor origin/main HEAD
git rev-list --left-right --count origin/main...HEAD
```

origin/main 前进时，用非破坏方式整合到功能分支并重跑受影响门禁。只普通推送功能分支：

```bash
git push -u origin feat/entity-relation-recall-community
```

不得推送 main，不得 force-push。网络、认证或沙箱阻塞时保留本地已验证提交并报告准确 push 命令，不把“未推送”说成“未完成代码”。

## 七、真正阻塞时的处理

只有以下情形可停在 `blocked`：

- 缺少必须由人提供的真实凭据或产品批准；
- 缺少足够人工 Gold，导致不能做 production promotion；
- 需要破坏性数据库/仓库操作而没有明确授权；
- 用户已有重叠修改，无法在不覆盖的情况下继续；
- 外部服务、网络或平台权限使对应 live 门禁无法运行，且所有独立离线工作已完成；
- 发现需求存在会导致架构完全不同的关键歧义。

阻塞前必须：

1. 留下最小复现、日志摘要和退出码；
2. 完成所有不依赖阻塞项的阶段；
3. 更新状态账本；
4. 给出一个明确问题和一条可直接恢复的命令；
5. 不伪造测试、Gold、模型调用或生产批准。

同一实现问题连续两轮修复仍失败时，记录根因假设和最小复现，可转做无依赖工作；但不得把依赖该门禁的后续阶段标为完成。

## 八、最终报告模板

最终报告必须以实际证据填写，不使用计划值冒充结果：

1. 开始/结束本地 HEAD、origin/main SHA、功能分支和远程分支 SHA；
2. baseline、corpus、Gold、policy、prompt、model、config、evaluator、graph input hash；
3. 实体和关系逐层漏斗的实际数量及守恒；
4. occurrence、admission、候选、归一、关系 pair、受控分类和失败恢复语义；
5. A/B 的 TP/FP/FN、Precision/Recall/F1、置信区间、candidate Recall、hard negatives、成本和时延；
6. treatment 是否晋级及原因，正式 control/P1/P2 是否保持不变；
7. GraphRAG 实际版本、API/CLI 命令、BYOG 字段映射、权重、自环、重复边、孤立点和空图；
8. fixture 节点、聚合边、层级、社区、成员及两次运行确定性比较；
9. community report 的迁移、表、run 生命周期、模型、快照绑定、引用与 live 结果；
10. Global Search 的 opt-in 路径、fail-closed、map/reduce、引用和 query Gold 结果；
11. 单元、完整 pytest、PostgreSQL、pip check、py_compile、Ruff、Mypy、迁移、Compose、Docker 的实际命令、退出码和测试数量；
12. 证明 P5 detection 未调用 LLM，以及 P6/P7 实际是否调用 fake/live 模型；
13. 证明 shadow 未读取不该读取的数据、未改变正式表、graph generation、现有查询、P1/P2 和社区 active 状态；
14. 修改文件、迁移、提交、推送、未跟踪文件保留和临时资源清理；
15. 未验证项、真实阻塞、下一命令。

最后逐项给出布尔状态，不得合并：

```yaml
engineering_complete: false
gold_gate_passed: false
shadow_isolation_passed: false
community_detection_passed: false
community_reports_live_verified: false
global_search_release_ready: false
production_approved: false
```

若只完成检测和报告工程闭环而未完成 Global Search/live/生产批准，准确表述为：

> GraphRAG BYOG 社区检测及 community reports 已形成 shadow 工程闭环；Global Search 尚未接入生产路径，社区结果尚未参与生产查询。

若 P7 工程也完成但未满足生产门禁，准确表述为：

> GraphRAG BYOG 社区检测、community reports 与 Global Search 已形成 opt-in shadow 工程闭环；社区结果尚未参与默认生产查询，生产启用仍受 Gold、live 模型和产品批准门禁约束。

全部可完成阶段完成前不要停止；若一次 Codex turn 结束，状态账本中的 `next_action` 必须让下一次 `resume` 可以直接接着实施。

## 九、Codex 命令参考

本文件中的启动、标准输入、JSONL 和 session resume 写法以当前 OpenAI 官方文档为准：

- [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Codex CLI command reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli#cli-codex-exec)
- [Codex execution plans guide](https://developers.openai.com/cookbook/articles/codex_exec_plans)
