# Knowledge-Forge Shadow 关系发现与可信发布专项

> Codex 阶段性实施、自检与验收任务书
> 版本：2026-07-24
> 适用仓库：`zjx2024/Knowledge-Forge`

## 0. 使用说明

本文是一份可直接交给 Codex 的完整实施任务书。Codex 必须先调查当前
HEAD，再按阶段顺序实施、自检和验证。历史审查基准
`main@cb0d833944e474bbae963b3c14bf4364b01aeb9f` 只用于定位问题，
不能 checkout、reset 或机械套用旧行号。

本任务的核心目标不是继续增加关系输出数量，而是：

```text
保留 Shadow 的高覆盖关系搜索
→ 把候选机会、分类假设、语义验证和发布事实彻底分层
→ 用证据、方向、Schema、去重和 freshness 将候选收敛为可信关系
→ 在同一人工 Gold 上证明最终关系召回是否真正提高
```

Codex 收到本文后必须：

1. 读取所有适用的 `AGENTS.md` 和仓库约定；
2. 检查当前 Git 状态并保护用户已有修改；
3. 先调查、复现并形成当前 HEAD 的执行计划；
4. 按 R0～R9 顺序连续实施；
5. R1～R7 先写失败测试再实施；R0、R8、R9 按只读调查/实验/验证规则执行；
6. 阶段门禁通过后自动进入下一阶段，不等待用户逐阶段确认；
7. 遇到本文定义的停止条件时立即停止，不得猜测或绕过；
8. 不 commit、不 push、不开 PR，除非用户之后明确授权。

本文不要求 Codex 修改本文本身。任务执行状态应在 Codex 的计划和最终报告中维护。

## 1. 已知实验事实与正确解释

当前实验使用相同的 2 份文档和 15 个分块：

| 指标 | 基线 | Shadow 增强 |
|---|---:|---:|
| 关系候选/结果 | 60 个候选 | 176 条 `classified` 关系假设 |
| 严格验证后可信关系 | 4 条并入图 | 尚未经过同等级 verifier |
| `related_to` | 较少 | 117 条，占 66.5% |
| 双向成对关系 | 少 | 26 组 |
| 完全重复边 | 少 | 约 10 条 |

已知错误示例：

- “汽车行业获得电池材料供应”被分类为“汽车行业生产电池材料”；
- 同一实体对经常同时产生 A→B 和 B→A；
- 电话与联系人被双向标为 `related_to`；
- 同一列表中的多个实体被错误标为 `related_to`；
- 部分 `uses`、`depends_on` 的方向与原文相反。

必须采用以下统计解释：

- 176 条是 `classified hypotheses`，不是 176 条候选机会的完整账本，也不是 176 条可信关系；
- 4 条是经过严格来源验证后可发布的关系；
- 176 与 4 不在同一流水线层级，不能直接比较；
- 没有完整 pair ledger 和人工 Gold 时，不能声称 Shadow 的关系召回率已经提高；
- 当前只能确认 Shadow 扩大了关系搜索范围和正关系假设数量；
- 本任务完成后，必须比较两套方案在同一 Gold 上的端到端严格 precision/recall。

任何代码、指标、接口或报告若继续把 `classified` 记为 `verified`，都属于 P0 缺陷。

## 2. 任务边界

### 2.1 本轮必须完成

- 冻结并可重放当前 Shadow 实验输入与结果；
- 建立完整、可审计的无序 mention pair opportunity ledger；
- 一个 pair 支持零到多条关系 hypothesis；
- 明确 `NO_RELATION`、`OTHER_UNSUPPORTED` 和具体关系假设；
- 仅在本次新建的 Shadow v2 candidate/release policy 中把 `related_to`
  覆盖为 review-only，不得自动发布；不得就地修改当前 active baseline
  policy 或共享 Schema；
- 实现 symmetric、inverse、canonical-forward 和自环规则；
- 增加独立 evidence verifier；
- verifier 验证精确 mention occurrence、证据、谓词和方向；
- 规范化、去重、反向冲突检测和多证据绑定；
- 严格区分 candidate、classified、verifier-supported、eligible 和 published；
- 修复跨实验事实比较、multiset 指标和发布门禁；
- 完善 fingerprint、freshness、stale 和数据库层 fail-closed；
- 建立人工 Gold 格式、校验器、分层评估和 pilot 重放；
- 保持正式基线和正式图不受 Shadow 写入影响。

### 2.2 本轮仅做最小保护，不做全面重构

本轮不同时重写完整实体抽取和实体 resolver。为隔离关系变量：

- 首轮重放必须冻结现有 mention occurrence 与 entity assignment；
- 关系候选必须基于精确 mention occurrence，而不是只基于 canonical 名称；
- 如果当前 HEAD 仍存在重复 canonical、错误 acronym 合并、alias 污染等实体问题，
  应加入执行报告和组合实验阻塞项；
- 只有直接阻塞关系阶段的端点问题才允许做最小修复，并必须单独给出测试证据；
- 大规模 alias provenance、实体 lease/CAS 和归一策略重构应留在独立任务中。

已知实体问题应保留为 characterization/组合实验夹具，至少包括：

```text
第三次出现 Canada 不得创建新的 canonical
lithium carbonate != lithium compounds
lithium chloride != lithium compounds
Cobalt acetates != Cobalt alloys
Cobalt carbonates != Cobalt cathode
Schema 类型通配符 "*" 正确工作
```

这些夹具用于暴露关系端点风险，但默认不是 relation
`engineering_complete` 的必过门禁。若它们在当前 HEAD 失败：

- 记录为 entity combination blocker；
- `production_approved=false`；
- 冻结关系实验必须继续使用明确的 assignment snapshot；
- 除非该缺陷直接阻断 R1～R7，不能借机扩大为完整实体重构。

若 Codex 能用严格最小、独立测试的修复解除直接阻塞，可在报告中把该修复与关系阶段分开；
否则留给实体专项。

### 2.3 明确不在当前 P0 中实施

- 不用 LightRAG 或 GraphRAG 直接替换权威事实层；
- 不接入新的图数据库；
- 不把 GraphRAG BYOG 当作关系抽取或验证器；
- 不把第二个候选模型的同意当作语义验证；
- 不因候选数量增加就宣布召回提高；
- 不在真实用户数据库、真实 HugeGraph 或生产 namespace 上做实验；
- 不修改生产 active policy 指针；
- 不激活当前已标记为 stale 的旧增强 snapshot。

## 3. 完成状态

Codex 最终必须区分以下状态，不能混用。

### 3.1 `engineering_complete`

满足：

- R0～R7 和 R9 的代码、兼容迁移、fake/disposable 测试和文档完成；
- 当前 176 条可按新状态机重放，或已明确记录缺失数据导致的不可重放字段；
- Shadow 对正式表零写入；
- 只有 fresh、`verifier_supported`、规范化后的关系可进入物化候选；
- 完整自动测试与静态检查通过；
- 真实外部模型、人工 Gold 或隔离服务缺失项被准确报告。

### 3.2 `pilot_evaluated`

除 `engineering_complete` 外，还满足：

- 2 文档、15 分块的输入、Schema、模型与策略被固定；
- 当前 176 条全部取得本次重放终态；
- 具备人工裁决的 pilot Gold；
- 基线与 Shadow 在同一 Gold、同一事实规范化规则下完成比较；
- 输出新增真实关系、精度、召回、方向和逐谓词指标。

### 3.3 `production_approved`

除 `pilot_evaluated` 外，还必须满足：

- Gold 规模、领域覆盖和独立复核达到仓库发布要求；
- 冻结 holdout 从未用于调参；
- 隔离环境真实模型、PostgreSQL、HugeGraph 和回滚演练通过；
- 质量门禁与成本门禁均通过；
- 有明确人工/运维批准。

当前 2 文档 pilot 无论结果多好，都不能单独形成 `production_approved=true`。

## 4. 永久不变量

### 4.1 事实与状态语义

必须保持以下语义分层：

| 层级 | 含义 | 是否可发布 |
|---|---|---|
| pair opportunity | 某两个精确 occurrence 值得判断 | 否 |
| classified hypothesis | classifier 提出的谓词、方向和证据假设 | 否 |
| verifier-supported fact | 独立 verifier 支持完整有向三元组 | 仍需规则与 freshness |
| materialization-eligible fact | 已通过规范化、去重、规则和 freshness | 可进入隔离 snapshot |
| published relation | active policy 下正式物化的事实 | 是 |

禁止使用一个模糊的 `verified` 字段覆盖多个层级。

### 4.2 Shadow 隔离

- Shadow 不写正式 `relation_assertions`、正式 projection、正式向量、outbox 或 HugeGraph；
- Shadow 不创建或修改正式 canonical entity、alias、assignment；
- Shadow 失败不改变产品图 generation 和状态；
- Shadow snapshot 只有在独立 gate 全部通过后才可成为候选 treatment；
- 本任务不得执行 treatment 激活；
- 数据库必须能够阻止绕过 service 层的非法激活。

### 4.3 Fail-closed

以下情况一律不可发布：

```text
非法 JSON
漏答或重复 pair_id
未知 pair_id
端点 ID 或 span 漂移
evidence 不可精确定位
evidence 未覆盖要求的 occurrence
未知 predicate
类型不兼容
方向不受证据支持
自环
分类/验证超时
验证 rejected
验证 uncertain
retryable_failed
OTHER_UNSUPPORTED
related_to review-only
旧策略或 stale snapshot
fingerprint 不完整
输入、Schema、模型、Prompt 或物化根不匹配
```

### 4.4 兼容性与迁移

- 数据库变化必须 additive、幂等且可升级；
- 不删除或篡改历史候选、事件、assertion 和审计数据；
- 新语义不能通过修改旧行策略版本来伪造历史验证；
- 旧读路径在迁移期间应兼容，或提供显式版本检测与安全失败；
- 新唯一约束必须先处理旧重复数据并保留审计映射；
- 所有重试必须幂等；
- 不持有数据库事务等待 LLM、Embedding、reranker 或 HugeGraph。

## 5. Git、工作区与安全规则

### 5.1 开始取证

Codex 在仓库根目录执行并记录：

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git rev-parse HEAD^{tree}
git status --short --branch
git diff --name-status
git diff --cached --name-status
git ls-files --others --exclude-standard
git diff --check
```

随后定位仓库约定：

```bash
find .. -name AGENTS.md -print
rg -n "pytest|ruff|mypy|pyright|black|isort|migration|alembic" \
  AGENTS.md pyproject.toml setup.cfg tox.ini Makefile .github 2>/dev/null
```

命令中不存在的文件可以跳过，但必须说明。

### 5.2 工作树保护

所有启动 Codex 前已有修改都属于用户。

禁止：

```text
git reset --hard
git clean
git stash
git restore
git checkout -- <path>
git add .
git add -A
git commit
git push
git push --force
git push --force-with-lease
```

若用户修改与本任务文件重叠：

1. 定位重叠 hunk；
2. 判断能否在不覆盖用户意图的情况下继续；
3. 能安全合并则保留双方变化；
4. 语义不明则触发停止条件并报告。

如需创建工作分支，必须在确认工作树状态后执行，失败时不得使用 `-f`：

```bash
git switch -c fix/shadow-relation-quality-v2
```

如果该分支已存在，使用唯一后缀，不删除或覆盖已有分支。

本文任务文件默认保持未跟踪，不纳入代码提交；Codex 只能将其作为执行约束读取。

### 5.3 外部资源

- 不打印 DSN、API key、token、真实正文或完整 Prompt 日志；
- 不自动安装 LightRAG、GraphRAG、GLiREL、REBEL 等新依赖；
- 不在用户真实数据库或图上运行迁移测试；
- 需要 disposable PostgreSQL 或隔离模型而环境未提供时，先完成 fake/单元测试并准确报告；
- 网络或权限失败不能用放宽 sandbox、`--yolo` 或危险命令绕过。

## 6. Codex 自动执行协议

每个阶段严格执行以下循环：

```text
1. 重新检查 git status 和本阶段允许修改范围
2. 读取当前实现和现有测试
3. 运行阶段相关现有测试，建立 before 基线
4. 添加 characterization 或修复前失败的回归测试
5. 证明测试失败原因与目标缺陷一致
6. 实施最小、兼容修复
7. 运行新增测试
8. 运行本阶段相关回归
9. 运行 git diff --check
10. 自审本阶段 diff、迁移、异常路径和状态机
11. 更新执行计划与阶段结果
12. 门禁通过后自动进入下一阶段
```

其中“先添加修复前失败测试”只适用于会改变行为的 R1～R7。R0 是只读调查，
R8 是冻结重放/实验，R9 是最终验证；这三个阶段不得为了满足形式要求而制造测试或修改算法。

R8 若缺少真实 176 数据、独立模型服务或人工 Gold，应记录：

```text
R8.status = BLOCKED_EXTERNAL
R8.blockers = [准确的缺失项]
pilot_evaluated = false
production_approved = false
```

随后继续执行 R9。只要 R0～R7、R9 的工程门禁通过，R8 的外部阻塞不妨碍
`engineering_complete=true`。不得伪造 R8 数据，也不得因等待外部输入无限运行。

不得：

- 先修改实现后补一个只会通过的测试；
- 删除、跳过或降低失败测试门槛；
- 用调高候选阈值隐藏结构缺陷；
- 通过截断候选提高 precision；
- 自动更新 Gold 使结果通过；
- 把环境缺失伪报成测试通过；
- 把未运行的命令写成 passed。

### 6.1 阶段停止条件

以下情况立即停止自动推进并报告：

- 当前工作树有无法判断归属的重叠修改；
- 需要破坏或迁移真实用户数据库/图；
- 需要新增对外产品语义但 Schema 无法唯一确定；
- 兼容迁移可能不可逆地丢失历史事实；
- 现有 failing test 与本任务无关且会掩盖阶段结果；
- 需要用户选择新的领域 predicate；
- 需要接受非商业许可证依赖；
- 需要真实人工 Gold 才能继续做生产批准。

人工 Gold 缺失只阻止 R8 的完整质量结论和生产批准，不阻止 R0～R7、R9 的工程实现。
若 R8 仅能完成 176 条规则重放、不能计算 Gold 指标，应把已完成部分和
`BLOCKED_EXTERNAL` 部分分别报告。

## 7. 目标状态机

Codex 应先映射当前表和枚举，再以最小兼容方式实现以下逻辑语义。
表名可按仓库风格调整，但语义不能减少。

```text
PAIR_PLANNED
  ├─ NO_RELATION
  ├─ OTHER_UNSUPPORTED
  ├─ CLASSIFIED_POSITIVE
  │    ├─ RULE_REJECTED
  │    ├─ VERIFIER_REJECTED
  │    ├─ VERIFIER_UNCERTAIN
  │    ├─ VERIFIER_RETRYABLE_FAILED
  │    └─ VERIFIER_SUPPORTED
  │          ├─ NORMALIZATION_REJECTED
  │          ├─ DUPLICATE_COLLAPSED
  │          ├─ REVIEW_ONLY
  │          └─ MATERIALIZATION_ELIGIBLE
  ├─ CLASSIFICATION_RETRYABLE_FAILED
  ├─ CLASSIFICATION_PERMANENT_FAILED
  └─ DEFERRED_BUDGET
```

要求：

- 每次 run 中每个 pair opportunity 都有状态；
- `DEFERRED_BUDGET` 是该 run 的显式终态，不等于已分类；
- 后续重试必须创建可审计 attempt，不静默覆盖历史；
- pair 级 `NO_RELATION` 和 `OTHER_UNSUPPORTED` 不创建正 hypothesis；
- 一个 pair 的多个 hypothesis 分别验证；
- 任一 hypothesis 的失败不得误改同 pair 其他 hypothesis；
- `VERIFIER_SUPPORTED` 不自动等于 published。

## 8. 阶段 R0：当前 HEAD 调查与可复现基线

### 8.1 目标

本阶段不修改代码。确认旧审查发现是否仍存在，并固定实际调用链。

### 8.2 必查代码

以当前 HEAD 为准，通过 `rg` 定位以下职责，不依赖旧行号：

```text
relation pair/candidate planning
candidate ID
classifier prompt/schema/parser
evidence/span checking
relation persistence
verification events
canonical projection/materialization
symmetric/inverse/domain/range schema
quality evaluator
release gate
snapshot fingerprint/freshness
experiment activate
graph generation/stale
```

历史上相关文件可能包括：

```text
knowledge_forge/relation_pipeline.py
knowledge_forge/database.py
knowledge_forge/quality.py
knowledge_forge/service.py
knowledge_forge/schema.py
数据库迁移文件
tests/ 下关系、质量、图状态和实体归一测试
```

### 8.3 必须回答

输出“已复现 / 未复现 / 证据不足”表：

1. 是否仍枚举 \(n(n-1)\) 个有向 mention pair；
2. candidate ID 是否包含有序端点；
3. `related_to` 是否为普通可发布 predicate；
4. 是否缺少 `OTHER_UNSUPPORTED`；
5. classifier 是否直接形成最终关系；
6. evidence 是否只检查字符串包含而不检查精确 occurrence；
7. verifier 是否验证完整 `(subject,predicate,object)`；
8. `classified` 是否被计入 verified 指标；
9. evaluator 事实键是否包含 snapshot/policy/内部 ID；
10. 是否用 `set` 隐藏重复；
11. stale snapshot 是否可能被 evaluate、gate 或 activate；
12. Shadow 是否存在正式写路径；
13. `"*"` 类型通配符是否正确；
14. 当前 176 条是否有完整 pair ledger；
15. `NO_RELATION`、失败、未执行和预算延期是否被保存。

### 8.4 基线测试

先从仓库配置确定 Python 和测试命令。若存在 `.venv/bin/python`，优先使用；
否则使用仓库规定的解释器。

至少运行：

```bash
python -m pytest -q <当前关系相关测试>
python -m pytest -q <当前质量与snapshot相关测试>
python -m pytest -q <当前图状态相关测试>
python -m py_compile knowledge_forge/*.py server.py
git diff --check
```

尖括号必须替换为真实路径，不能原样执行。若仓库没有 `server.py`，按实际入口调整。

### 8.5 R0 门禁

- 没有修改任何文件；
- 已记录 HEAD、工作树和实际命令；
- 每个已知问题都有当前代码证据；
- 已确认现有测试缺口；
- 已形成 R1～R9 的实际文件级计划。

## 9. 阶段 R1：冻结实验、统一语义与分层指标

### 9.1 目标

先修正“如何计数和比较”，再改变算法。否则后续实验仍不可解释。

### 9.2 冻结内容

在不泄露真实敏感正文的前提下，为现有 pilot 保存或引用：

```text
document revision/hash
15 个 chunk 的稳定清单与 hash
mention occurrence ID、绝对 span 和 assignment version
Schema 与 predicate 元数据
candidate/classifier/verifier policy
prompt 与 model revision
预算和阈值
176 条 classified hypotheses
每条 hypothesis 的 pair、端点、谓词、方向、证据和原状态
旧 snapshot/fingerprint
```

若数据库中不存在完整 pair ledger：

- 不伪造历史 `NO_RELATION`；
- 不从 176 条反推候选总数；
- 报告 `candidate_recall = unavailable`；
- 新代码从下一 run 开始记录完整账本。

真实正文、完整模型响应和敏感数据不得提交为测试 fixture。使用合成最小文本复现错误；
真实 pilot 通过本地/隔离评估输入引用。

### 9.3 指标语义

漏斗至少报告：

```text
pair_opportunities_total
pair_opportunities_classified
pair_no_relation
pair_other_unsupported
pair_deferred_budget
pair_failed
hypotheses_total
hypotheses_rule_valid
hypotheses_verifier_supported
hypotheses_verifier_rejected
hypotheses_verifier_uncertain
hypotheses_retryable_failed
facts_after_normalization
facts_duplicate_collapsed
facts_review_only
facts_materialization_eligible
facts_published
```

不得出现：

```text
verified = classified count
recall = positive output count
precision = supported count / published count（若未标注 Gold）
```

### 9.4 两级评价键

跨实验语义事实键和 occurrence assertion 键都不得包含：

```text
snapshot_id
policy_hash
内部 candidate/relation ID
run_id
数据库行 ID
```

必须分别维护：

```text
canonical fact key:
  project/Gold universe
  aligned subject canonical identity
  canonical predicate
  aligned object canonical identity
  canonical direction

occurrence assertion key:
  canonical fact key
  document revision or Gold document key
  subject occurrence/span
  object occurrence/span
  Gold evidence span identity
```

`canonical fact key` 用于比较“发现了哪些事实”，不得包含 evidence 或 provenance。
`occurrence assertion key` 用于严格证据、方向和提及级评估。同一语义事实的
provenance 作为附属集合比较，不得反过来污染 canonical fact 身份。

评价必须使用 multiset，不能先 `set()` 后隐藏完全重复。

### 9.5 回归测试

至少新增：

- 10 条 `classified`、2 条 supported 时，`verifier_supported` 只能是 2；
- 同一事实跨 snapshot/policy 仍能匹配；
- 两条完全重复输入在 multiset 中计为 duplicate；
- 反向 pair 不得被 set 隐藏；在 verifier 后区分
  symmetric/inverse duplicate、unsupported reverse conflict 和 verified reciprocal；
- 没有 Gold 时 recall 字段为 unavailable/拒绝计算；
- stale 旧实验不能作为生产评估通过证据。

### 9.6 R1 门禁

- `classified`、`supported`、`eligible`、`published` 统计完全分离；
- 当前 176 条被明确标记为 hypotheses；
- 报告不再声称“176 对 4 = 召回提高”；
- 旧行为兼容读取；
- 相关测试通过。

## 10. 阶段 R2：无序 mention pair opportunity ledger

### 10.1 目标

每个精确 occurrence pair 只获得一次“是否存在关系”的判断机会，方向在 hypothesis
阶段决定，不再分别生成 A→B 和 B→A 两个候选。

### 10.2 Pair 身份

pair opportunity 必须保留：

```text
project_id
document_revision_id
left mention occurrence stable key
right mention occurrence stable key
left/right exact absolute span
left/right canonical assignment version
context window identity/span
candidate policy version
allowed predicate metadata
planning reason/cue
attempt/run provenance
```

稳定身份与运行级机会 ID 必须分开。跨 candidate policy 比较使用：

```text
pair_identity = hash(
  document_revision,
  min(occurrence_key_a, occurrence_key_b),
  max(occurrence_key_a, occurrence_key_b)
)
```

具体规划记录使用：

```text
opportunity_identity = (
  run_or_snapshot,
  candidate_policy_version,
  pair_identity
)
```

candidate policy、context、cue、预算和 attempt 只属于 planning provenance，
不得改变跨策略稳定 `pair_identity`。不得使用实体表面词作为唯一身份。
重叠窗口发现同一 pair 时必须幂等合并或选择确定性 context；同一
run/policy 不能产生两份 opportunity。

### 10.3 候选规划

- 同句、相邻句、字符距离、cue 和类型规则继续保持高覆盖；
- 不得用更严格阈值换 precision；
- 不得 `candidates[:N]` 静默裁剪；
- 超预算项记录为 `DEFERRED_BUDGET`；
- 同 canonical 自环默认在计划或硬规则阶段过滤并记录原因；
- “类型未知”与“类型不兼容”必须分开；
- Schema 中 `"*"` 必须正确表示通配；
- \(n\) 个同窗口 occurrence 的理论 pair 上限是 \(n(n-1)/2\)，不是 \(n(n-1)\)；
- 真实局部窗口可能减少 pair，但必须以 coverage/Gold 门禁证明没有静默漏候选。

### 10.4 一个 pair 多个关系

数据模型必须允许：

```text
{A,B}
  → A produces B
  → A supplies B
```

如果原文同时支持两个具体关系，两个 hypothesis 分别保存和验证。不得因为 pair 唯一就限制为一条边。

### 10.5 回归测试

至少覆盖：

- 2 个合格 occurrence 只生成 1 个 pair；
- 3 个 occurrence 最多生成 3 个 pair，而不是 6 个；
- 输入实体顺序反转时 `pair_identity` 不变；
- candidate policy 变化时 `pair_identity` 不变，但 opportunity provenance 不同；
- 重叠窗口不重复 pair；
- 相同表面词但不同 occurrence 不被错误折叠；
- 同 canonical 自环被过滤；
- wildcard 类型规则正确；
- unknown type 不被当作 incompatible；
- 预算不足有显式状态；
- 一个 pair 后续可关联两个 hypothesis。

### 10.6 R2 门禁

- A→B/B→A 双候选来源被结构性消除；
- 所有 pair 有可审计状态；
- candidate coverage 在现有 fixture 上不低于旧 Shadow；
- classifier、verifier 和发布逻辑尚未被混入本阶段；
- 数据库迁移兼容并通过测试。

## 11. 阶段 R3：关系分类契约与 Schema 方向语义

### 11.1 目标

classifier 只提出可验证的关系假设，不作最终发布裁决。

### 11.2 输出契约

每个 pair 必须且只能返回以下三类之一。方向只有一个事实来源：
`subject_occurrence_id` 与 `object_occurrence_id` 的有序角色。classifier
持久化契约不得再保存可与端点角色冲突的独立 `direction` 字段；若为了兼容
旧记录必须读取该字段，它只能由端点角色派生，任何不一致都必须 fail-closed。
symmetric predicate 仍填写确定性的 subject/object，后续由 Schema 的
`symmetric=true` 规范化，不能使用第二套方向真相。

```json
{
  "pair_id": "stable-pair-id",
  "decision": "NO_RELATION",
  "hypotheses": []
}
```

```json
{
  "pair_id": "stable-pair-id",
  "decision": "OTHER_UNSUPPORTED",
  "hypotheses": [],
  "evidence_spans": [{"start": 0, "end": 10}],
  "reason_code": "SCHEMA_GAP"
}
```

```json
{
  "pair_id": "stable-pair-id",
  "decision": "HYPOTHESES",
  "hypotheses": [
    {
      "subject_occurrence_id": "mention-a",
      "predicate": "uses",
      "object_occurrence_id": "mention-b",
      "evidence_spans": [{"start": 0, "end": 30}],
      "classifier_confidence": 0.82
    }
  ]
}
```

字段名可按仓库风格调整，但语义必须等价。

### 11.3 决策定义

- `NO_RELATION`：原文只共现、同列、同表、同联系人块，或没有明确语义关系；
- `OTHER_UNSUPPORTED`：原文明确存在关系，但当前 Schema 没有合适具体 predicate；
- `HYPOTHESES`：存在一个或多个当前 Schema 可表达的具体关系；
- classifier 不能因必须回答而强制输出正关系；
- `related_to` 不得作为 Schema 缺口兜底；
- `related_to` 的 review-only 规则是 Shadow v2 candidate/release policy
  覆盖，不得原地修改 active baseline policy 或共享 Schema；未来若要正式采用，
  必须创建新版本 policy，并经过完整 gate/promotion；
- 在 Shadow v2 中，`related_to` 即使保留，也必须是 review-only、不可自动发布。

### 11.4 Schema 元数据

每个 predicate 应明确：

```text
domain
range
symmetric
inverse predicate（若有）
canonical-forward rule
publishable/review-only
description与正反例
schema version
```

规则：

- symmetric predicate 的语义不依赖左右方向；
- inverse 必须可映射到统一 canonical predicate/direction；
- canonical-forward 只负责规范表示，不能创造证据未支持的方向；
- 同 domain/range 不表示 predicate 对称；
- 一个 pair 可以输出多个不同 predicate。

### 11.5 Prompt 要求

Prompt 必须明确给出 hard negative：

- 同一列表出现本身不等于关系；若列表结构或正文明确表达具体谓词，仍应保留；
- 电话号码与联系人字段共现不自动形成 `related_to`；明确的
  `has_contact_number` 等 Schema 事实可正常提出并验证；
- “获得供应”不等于“生产”；
- A 使用 B 与 B 使用 A 不同；
- A 依赖 B 与 B 依赖 A 不同；
- Schema 无谓词时选择 `OTHER_UNSUPPORTED`；
- 无语义关系时选择 `NO_RELATION`。

Prompt、JSON Schema、parser 和模型配置必须独立版本化并进入 fingerprint。

### 11.6 Parser 完整性

对每个 batch：

- 输入 pair ID 必须恰好被回答一次；
- 未知、重复、漏答 ID 整批或对应项 fail-closed；
- occurrence ID 必须属于该 pair；
- subject/object 的有序角色是方向唯一真相；兼容读取的旧 `direction` 若与
  端点角色或 predicate 的 symmetric/inverse 元数据冲突，必须 fail-closed；
- evidence span 必须落在冻结 context；
- predicate 必须属于该 pair 的 allowed set；
- 决策与 hypotheses 数量一致；
- confidence 只作分析信息，不赋予发布资格。

### 11.7 回归测试

至少覆盖：

- 同列表且没有明确语义关系 → `NO_RELATION`；
- 电话与联系人仅结构共现、没有具体谓词 → `NO_RELATION`；
- 明确关系但 Schema 缺词 → `OTHER_UNSUPPORTED`；
- 一个 pair 两个合法 predicate；
- left-to-right 与 right-to-left；
- symmetric（不得引入独立且可能冲突的 direction 字段）；
- inverse；
- `related_to` review-only；
- 未知 predicate；
- 端点漂移；
- 重复/漏答 pair ID；
- evidence 越界；
- 非法 JSON、超时和 repair 失败。

### 11.8 R3 门禁

- classifier 不再直接产出可发布 assertion；
- 所有输出都可追溯到 pair 和 occurrence；
- `related_to` 自动发布数恒为 0；
- `NO_RELATION` 与 `OTHER_UNSUPPORTED` 语义清晰；
- 多 hypothesis、方向和 Schema 规则测试通过。

## 12. 阶段 R4：确定性硬校验与独立 evidence verifier

### 12.1 目标

把“模型认为可能有关系”与“原文确实支持完整有向三元组”分开。

### 12.2 验证顺序

```text
classified hypothesis
→ deterministic validation
→ independent semantic verifier
→ verifier verdict
```

确定性校验先检查：

- pair 和 occurrence 身份；
- absolute/local span 映射；
- evidence 边界；
- evidence 是否覆盖两个要求的 occurrence；
- predicate 是否存在；
- domain/range；
- 类型未知不得等同于类型不兼容，但也不得默认通过：若 predicate 的
  domain/range 包含 `"*"`，或当前 snapshot 中的正向类型证据确认兼容，可继续；
  否则终态只能是 `TYPE_UNRESOLVED/REVIEW_ONLY`，不可获得
  materialization eligibility；
- self-loop；
- symmetric/inverse 元数据完整性；
- review-only；
- stale/fingerprint 基础字段。

未通过硬校验的 hypothesis 不调用 verifier，记录明确 reason code。

### 12.3 Verifier 输入

每次 verifier 判断一个具体 hypothesis，输入至少包括：

```text
冻结原文 context
两个精确 occurrence 及 span
完整 subject-predicate-object
predicate 定义
允许的方向和类型约束
classifier 提供的 evidence span
```

Verifier 不得：

- 依赖外部知识补全；
- 因两个实体相关就支持具体 predicate；
- 只判断无序 `{A,B,predicate}`；
- 把 classifier confidence 当作验证结果；
- 复用 classifier positive 标记作为 supported。

独立意味着必须有独立判定调用、独立 prompt/version 和独立事件。
底层可暂时使用同一个模型服务，但必须记录模型 revision，并在评估中披露；
不能假装同模型自证等于人工 Gold。

### 12.4 Verifier 输出

```json
{
  "hypothesis_id": "stable-id",
  "verdict": "SUPPORTED",
  "reason_code": "DIRECTLY_ENTAILED",
  "supported_evidence_spans": [{"start": 0, "end": 30}]
}
```

合法 verdict：

```text
SUPPORTED
REJECTED
UNCERTAIN
RETRYABLE_FAILED
```

只有 `SUPPORTED` 可进入 R5。`UNCERTAIN` 不能默认通过。

### 12.5 必须验证的语义

- 证据直接支持该 predicate，而非近义猜测；
- subject 与 object 角色正确；
- 方向正确；
- 否定、假设、计划、比较和引用语境不被误当事实；
- evidence 与候选 occurrence 对齐；
- 类型约束成立；
- 多 span 或跨句证据保持可定位。

### 12.6 回归测试

必须包含合成最小文本：

| 场景 | 预期 |
|---|---|
| 汽车行业获得电池材料供应 | `automotive_industry produces battery_material` 被拒绝 |
| A uses B | B→uses→A 被拒绝 |
| A depends_on B | B→depends_on→A 被拒绝 |
| A、B 仅在同一列表且没有明确语义 | 正关系被拒绝 |
| 电话与联系人仅结构共现 | `related_to` 不可支持/发布 |
| evidence 只覆盖一个端点 | 拒绝 |
| evidence 指向同名的另一 occurrence | 拒绝 |
| 端点类型未知且 predicate 无 `"*"`、无正向兼容证据 | `TYPE_UNRESOLVED/REVIEW_ONLY`，不得发布 |
| 端点类型未知但 predicate 对应端为 `"*"` | 可继续 verifier，仍须满足其他全部门禁 |
| 明确否定关系 | 拒绝 |
| 明确假设关系 | 按 Schema 事实规则拒绝或 uncertain |
| verifier 超时 | `RETRYABLE_FAILED` |
| verifier 漏答/非法 JSON | fail-closed |
| 同 pair 两个关系，仅一个有证据 | 分别 supported/rejected |

### 12.7 R4 门禁

- classifier positive 不再具备发布资格；
- 只有 verifier `SUPPORTED` 可进入下一阶段；
- 方向、谓词和精确 span 均被独立验证；
- unknown type 仅可作为候选进入 verifier；未由 `"*"` 或当前 snapshot
  正向类型证据解决时，始终 review-only、不可发布；
- 失败与 uncertain 全部 fail-closed；
- 基线严格 verifier 的已有行为不被放宽。

## 13. 阶段 R5：规范方向、去重、多证据与幂等持久化

### 13.1 目标

先把受支持 hypothesis 转成统一语义事实，再去重。去重不能代替方向验证。

### 13.2 规范化顺序

```text
verifier-supported hypothesis
→ inverse 转换
→ symmetric/canonical-forward 端点规范
→ self-loop/方向冲突检查
→ semantic fact key
→ duplicate collapse
→ provenance/evidence 绑定
```

例如若 Schema 定义：

```text
is_supplied_by inverse_of supplies
```

则：

```text
B is_supplied_by A
```

规范为：

```text
A supplies B
```

不能把没有 inverse 元数据的两个反向谓词自行视为相同。

### 13.3 事实与证据分离

语义事实身份建议由以下信息构成：

```text
project_id
normalized subject canonical identity
canonical predicate
normalized object canonical identity
```

snapshot、policy、fingerprint 和 run 作为该事实的版本化 membership/provenance，
不得写进跨实验 semantic fact key。

证据身份独立包含：

```text
document revision
subject/object occurrence
evidence span(s)
classifier/verifier events
policy/model provenance
```

同一事实由多个文档或 span 支持时：

- 只保留一个 canonical fact/projection；
- 保存多个不可变 evidence/assertion；
- 不丢失每条 occurrence-level 审计；
- evaluator 可以分别计算 fact-level 和 evidence-level 指标。

### 13.4 冲突规则

- 完全重复：折叠为一个事实，多份 provenance；
- symmetric 反向重复：规范端点后折叠；
- inverse 重复：转换后折叠；
- 非 symmetric、无 inverse 的 A→P→B 与 B→P→A：
  不能仅因反向就自动折叠；若两条都分别有明确、独立的 verifier-supported
  证据，则作为合法 reciprocal facts 保留；否则标记 unsupported reverse conflict/review；
- 同一 pair 的不同合法 predicate：不得误删；
- 自环：默认拒绝，除非 Schema 显式允许 reflexive。

### 13.5 数据库与幂等

- 为 pair、hypothesis、verification attempt、normalized fact 和 evidence 建立清晰唯一键；
- 重试不得重复写事实或 evidence；
- lease 重领/进程恢复不得产生第二条相同事实；
- candidate/event/assertion 的事务边界保持现有审计不变量；
- 模型调用在事务外；
- migration 必须先扫描/报告旧重复，再安全回填；
- 历史重复不能被无记录删除，应保存 collapse/redirect 审计。

### 13.6 回归测试

至少覆盖：

- 完全重复 collapse；
- symmetric 反向 collapse；
- inverse collapse；
- 非对称、双边均有证据的 reciprocal facts 被保留；
- 只有一边受支持的 reverse conflict 不双边发布；
- 多 evidence 绑定同一事实；
- 同 pair 多 predicate 保留；
- 自环拒绝；
- 同一 batch 重放幂等；
- lease/retry 幂等；
- migration 在含旧重复数据时可升级；
- 旧读路径兼容。

### 13.7 R5 门禁

在测试数据上：

```text
完全重复事实 = 0
symmetric/inverse 规范化重复 = 0
unsupported reverse conflict 发布 = 0
非法自环 = 0
合法多谓词保留率 = 100%
verified reciprocal facts 保留率 = 100%
每个 materialization-eligible fact 均有至少一条 verifier-supported evidence
```

## 14. 阶段 R6：物化边界、fingerprint、freshness 与 stale

### 14.1 目标

保证正确但过期的结果也不能发布。

### 14.2 Fingerprint 最小覆盖

必须确定性覆盖：

```text
source generation
document revision/hash
chunk universe 与每个 chunk hash
mention occurrence universe
entity assignment version/snapshot
Schema 及 predicate 元数据
candidate policy/version/参数/预算
classifier prompt/schema/parser version
classifier model与revision
verifier prompt/schema/parser version
verifier model与revision
所有影响行为的阈值
normalization/symmetric/inverse规则
publishable/review-only规则
quality gate policy/version/noninferiority delta/置信方法/最小样本
代码/策略 materialization root
```

不得只记录一个笼统 `policy_hash` 而无法解释组成。

### 14.3 Freshness 检查点

以下路径都必须独立检查，不得只依赖上层 service：

```text
evaluate
quality gate
treatment snapshot completion
activate
materialize
database CAS/update
graph rebuild input
```

任一 fingerprint 组成不匹配：

- snapshot 转为或被视为 stale；
- evaluate 可输出历史分析，但不得作为当前 release gate；
- gate 失败；
- activate 失败；
- materialize/publish 失败；
- 不能人工修改状态绕过。

### 14.4 Shadow 零写入

增加数据库级测试证明 Shadow 不会：

- 插入正式 assertion/projection；
- 写正式 relation vector/outbox；
- 写 HugeGraph；
- 变更 active policy；
- 递增正式 graph generation；
- 把 graph 从 stale 改为 ready；
- 修改正式 entity/alias/assignment。

### 14.5 发布谓词

只有同时满足以下条件的 normalized fact 才具有 materialization eligibility：

```text
classifier concrete hypothesis
deterministic validation pass
verifier SUPPORTED
not review-only
not duplicate conflict
not self-loop
current fingerprint
complete evidence
current assignment/schema
type compatible in current snapshot, or applicable domain/range is "*"
snapshot complete
```

`related_to`、`OTHER_UNSUPPORTED`、`TYPE_UNRESOLVED`、`UNCERTAIN`
和失败项永远不满足。此处的 `related_to` 限制属于 Shadow v2
candidate/release policy；不得因此改变当前 active baseline。

### 14.6 回归测试

逐一改变以下字段并证明旧 snapshot stale：

- source generation；
- 一个 chunk hash；
- entity assignment；
- Schema；
- symmetric/inverse 元数据；
- classifier prompt；
- classifier model revision；
- verifier prompt；
- verifier model revision；
- candidate budget；
- threshold；
- normalization rule；
- materialization root。

另测：

- service 层拒绝 activate；
- 直接 database activate/CAS 同样拒绝；
- 不完整 snapshot 拒绝；
- 混合新旧 policy 拒绝；
- stale 结果无法进入图。

### 14.7 R6 门禁

- evaluate、gate、activate、materialize 和 DB 层均 fail-closed；
- Shadow 正式写入数为 0；
- 旧 176 snapshot 不可直接激活；
- 所有 freshness 测试通过。

## 15. 阶段 R7：人工 Gold、分层评价器与发布门禁

### 15.1 Gold 原则

Gold 必须由人工标注或独立人工裁决。模型可以生成候选标注包，但：

- 模型自产标签不得标为 human Gold；
- 同一个模型不能生成 Gold 后评价自身；
- dev 与 holdout 必须分离；
- holdout 内容 hash 和 manifest 冻结；
- 标签修改需要版本和 reviewer 记录；
- 无有效 Gold 时，系统拒绝给出“召回提高”的结论。

### 15.2 Gold 格式

Gold 必须拆成 pair-level decision 与 fact-level positive，不得让
`NO_RELATION`/`OTHER_UNSUPPORTED` 被迫填写并不存在的 predicate。

每个 pair-level Gold 至少包含：

```text
dataset/version/split
document revision/hash
left/right exact occurrence/span
left/right canonical Gold identity
decision: NO_RELATION | OTHER_UNSUPPORTED | HAS_RELATION
evidence span(s)
unsupported relation description（仅 OTHER_UNSUPPORTED，可选）
critical flag
annotator/reviewer
reviewed_at
adjudication status
```

每个 `HAS_RELATION` pair 关联一到多个 fact-level Gold：

```text
pair identity
subject exact occurrence/canonical Gold identity
predicate
object exact occurrence/canonical Gold identity
direction
evidence span(s)
critical flag
adjudication status
```

`NO_RELATION` 不得有正 fact；`OTHER_UNSUPPORTED` 可以有关系描述和 evidence，
但不要求伪造当前 Schema predicate。

必须定义冻结的 adjudication universe：

- 计算 recall 时，冻结文档范围内的所有 Gold 正事实必须完成穷尽标注；
- 计算 precision 时，control 或任一 treatment 预测出的所有事实必须已被人工裁决；
- 未裁决预测不得默认计为 false positive，也不得默认计为 true positive；
- 只要预测集合存在未裁决项，precision 标记为 unavailable，release gate fail-closed；
- 可以另报“已裁决子集 precision”，但不能用它批准生产。

Hard negatives 必须覆盖：

```text
同列表且没有明确语义关系
同表格但无语义关系
电话/联系人仅结构共现且没有具体谓词
供应 vs 生产
uses 反向
depends_on 反向
否定
假设
比较
引用他人观点
同名不同 occurrence
多谓词
Schema 缺词
related_to 兜底
```

### 15.3 分层指标

必须分别报告：

1. endpoint recall：Gold 关系两端 occurrence 是否都存在；
2. pair coverage：Gold pair 是否进入无序 opportunity ledger；
3. hypothesis recall：classifier 是否提出正确 predicate 和方向；
4. verifier recall：正确 hypothesis 是否被 supported；
5. final strict precision/recall/F1；
6. predicate macro/micro P/R/F1；
7. direction accuracy；
8. exact evidence coverage；
9. `NO_RELATION` precision/recall；
10. `OTHER_UNSUPPORTED` precision/recall；
11. `related_to` classified/review/published 占比；
12. 完全重复、symmetric/inverse 重复、unsupported reverse conflict、
    verified reciprocal facts 和自环；
13. 相对基线新增且 Gold 支持的事实；
14. 基线 Gold 事实保留率；
15. 每个阶段的调用、token、延迟和失败率。

### 15.4 比较口径

基线与 Shadow 必须使用：

```text
同一 Gold
同一 document revision/chunk universe
同一 entity alignment
同一 Schema
同一 strict fact normalization
同一 evidence 规则
```

若其中一项不同，报告必须明确说明不可直接归因。

### 15.5 发布硬门禁

最低门禁：

```text
stale 发布 = 0
未验证发布 = 0
invalid evidence 发布 = 0
related_to 自动发布 = 0
OTHER_UNSUPPORTED 发布 = 0
完全重复 = 0
symmetric/inverse 规范化重复 = 0
unsupported reverse conflict 发布 = 0
非法自环 = 0
cannot-link violation = 0
发布关系 evidence 精确可定位 = 100%
发布关系 verifier-supported = 100%
所有 pair 在本次 run 有状态 = 100%
```

质量门禁：

- 在同一冻结 occurrence universe 上，Shadow 的唯一无序 pair coverage 和
  Gold pair coverage 不低于旧 Shadow；不得拿旧版有向候选原始条数与新版
  无序 pair 数量直接比较；
- 基线 4 条严格支持事实应被新链路保留，除非人工 Gold 证明基线本身错误；
- Shadow strict precision 非劣门槛必须来自版本化 release policy，例如
  `relation_precision_noninferiority_delta`、最小 Gold 数量和置信区间方法；
  未配置时 gate fail-closed，Codex 不得自行发明 δ；
- Shadow Gold end-to-end recall 必须实际高于基线，才能声称达到“增强关系发现”目标；
- 当前 pilot 样本太小时，不报告虚假的小数优势，改为逐例新增真实关系和零新增关键错误；
- 最终门禁必须单独检查 precision，不能只用 F1 掩盖 precision 下降。

如果工程门禁全部通过但 Gold recall 没有提高，应报告：

```text
engineering_complete = true
goal_met = false
production_approved = false
```

这是一项有效实验结论，不得通过无界调参或反复改 Gold 强行达到收益目标。

### 15.6 小样本解释

2 文档 pilot 可支持：

```text
发现具体新增真实关系
识别错误类型
证明状态机与门禁工作
验证基线事实是否保留
比较逐例方向和谓词
```

不能支持：

```text
总体生产质量优于基线
跨领域稳定提升
百分点级统计显著性
替换正式方案
```

### 15.7 R7 门禁

- Gold manifest 和导入校验存在；
- evaluator 不会在无 Gold 时输出 recall；
- multiset 不隐藏重复；
- 高召回低精度 treatment 无法通过；
- 所有质量与 freshness 门禁有回归测试。

## 16. 阶段 R8：冻结重放与同输入 pilot

本阶段不再修改算法。若发现算法缺陷，回到对应阶段修复并重新运行相关回归。

### 16.1 第一步：重放当前 176 条

不重新生成候选，先把当前 176 条 hypotheses 送入：

```text
deterministic validation
→ independent verifier
→ normalization
→ duplicate/conflict handling
```

输出漏斗：

```text
176 classified
→ rule-valid
→ verifier-supported/rejected/uncertain/failed
→ normalized facts
→ duplicate collapsed
→ review-only
→ materialization-eligible
```

对每条 rejected/review 项记录 reason code。重点错误簇：

```text
related_to
供应/生产
uses 方向
depends_on 方向
同列表
电话/联系人
完全重复
反向关系（区分规范化重复、unsupported conflict、verified reciprocal）
```

如果旧 176 条缺少精确 occurrence 或 evidence：

- 不猜测对齐；
- 进入 `UNALIGNED`/rule rejected；
- 报告旧数据不可进行严格重放的比例；
- 不修改旧数据伪造 span。

### 16.2 第二步：运行新完整 Shadow

固定：

```text
相同 2 文档 revision
相同 15 chunk
相同 entity mention/assignment snapshot
相同 Schema
固定 classifier/verifier model revision
固定 prompts 和 budgets
```

分别运行：

```text
Control baseline
Native Shadow v2
```

在同一 Gold 上比较 R7 全部指标。

### 16.3 必须回答

1. 176 条中多少被 verifier 支持；
2. 规范化去重后剩多少可信事实；
3. 117 条 `related_to` 中多少仅为共现；
4. 26 组双向关系中多少是候选结构导致；
5. 是否保留基线 4 条；
6. 新增多少 Gold 支持事实；
7. 新增关系来自哪些 pair/predicate；
8. Shadow 最终 precision 是否达标；
9. Shadow 最终 recall 是否实际提高；
10. 成本增加来自 candidate、classifier 还是 verifier。

### 16.4 R8 门禁

- 176 条全部有本次重放状态，或明确列出无法对齐项；
- 旧 stale snapshot 未被激活；
- 结果报告不混淆层级；
- 无人工 Gold 时只报告漏斗，不声称召回提高；
- pilot 不自动改变 production active policy。

## 17. 阶段 R9：完整回归、迁移验证、自审与最终报告

### 17.1 完整验证

从仓库配置读取真实命令，运行：

```text
新增 targeted tests
全部关系 tests
实体端点最小回归
数据库迁移升级 tests
质量与 snapshot tests
图状态与发布 tests
完整 pytest
py_compile
仓库已有 lint/format/type checks
git diff --check
```

如果仓库未配置 lint/type checker，如实说明，不臆造新工具。

如果存在 disposable PostgreSQL 环境变量，只在明确是测试库时运行集成测试；
不得打印 DSN。

### 17.2 最终自审

逐项检查：

- 是否仍生成 A→B 和 B→A 两份 pair opportunity；
- 无序 pair 是否保留精确 occurrence 和最终方向；
- 一个 pair 是否支持多个合法 predicate；
- `NO_RELATION` 与 `OTHER_UNSUPPORTED` 是否混淆；
- `related_to` 是否存在任何自动发布路径；
- classifier 是否可能绕过 verifier；
- verifier 是否检查完整有向三元组；
- evidence 是否可能对齐同名错误 occurrence；
- rejected、uncertain、failed 是否可能物化；
- inverse/symmetric 是否可能误删合法事实；
- 完全重复、非法反向冲突、合法 reciprocal 和自环是否可见；
- classified 是否仍被称为 verified；
- evaluator 是否混入 provenance 或用 set 隐藏重复；
- stale 是否能绕过 evaluate/gate/activate/DB；
- Shadow 是否能写正式表；
- migration 是否幂等、兼容并保留历史；
- fingerprint 是否遗漏任何行为控制量；
- 报告是否仍以输出数量代替 recall。

### 17.3 最终报告

Codex 用中文输出：

1. 当前 HEAD、分支和开始/结束工作树；
2. R0～R9 每阶段状态；
3. 修改文件和职责；
4. 数据库迁移与兼容策略；
5. 每条测试命令及 passed/skipped/failed；
6. 176 条重放漏斗；
7. 基线与 Shadow 的同层指标；
8. 完全重复、symmetric/inverse 重复、unsupported reverse conflict、
   verified reciprocal、自环和 `related_to` 指标；
9. 新增 Gold 真实关系逐例清单；
10. 未验证环境和剩余风险；
11. `engineering_complete`、`pilot_evaluated`、`production_approved` 三个布尔值及证据；
12. 明确确认未 commit、未 push、未修改正式 active policy。

### 17.4 R9 门禁

- 完整回归无未解释失败；
- `git diff --check` 通过；
- 所有 skipped 有真实原因；
- 无凭据、真实语料、dump 或临时大文件进入 diff；
- 最终报告与实际命令输出一致。

## 18. 调研报告的采用边界

用户调研报告提出 LightRAG、GraphRAG、GLiNER/GLiREL、REBEL、Splink 等方案。
本任务采用以下结论：

### 18.1 直接采用

- Knowledge-Forge 原生 mention-level、evidence-level、可回滚事实层继续作为权威层；
- LightRAG 若将来使用，只能作为隔离 sidecar 或候选/检索对照；
- GraphRAG 若将来使用，只能消费已经 verifier-supported、规范化和去重的图；
- 外部候选器不能直接写正式 assertion、canonical 或 HugeGraph；
- 所有候选源必须使用同一 verifier、Gold 和发布门禁。

### 18.2 当前不采用

- 不在本轮接入 LightRAG；
- 不在本轮实现 GraphRAG BYOG；
- 不把 LightRAG `import_native_kg` 当作关系发现增强，它只是导入已有关系；
- 不假设 LightRAG 输出原生具备 Knowledge-Forge 所需 exact character span；
- 不把 GLiREL/REBEL 当作 verifier；
- 不直接执行报告中的 `pip install` 和示例脚本；
- 不使用未经实测的人日、GPU 和质量收益估计作为实施依据。

### 18.3 原因

当前瓶颈是：

```text
classified 未经同等级验证
related_to 兜底
方向错误
双向/完全重复
证据 span 不严格
状态和 freshness 门禁缺失
```

增加另一个候选生成器不会自动解决这些问题，反而会把更多噪声送入尚未修好的链路。

此外：

- GraphRAG BYOG 官方设计本来就假定实体和关系抽取已完成，适合下游总结；
- GLiREL 官方示例可能对同一 pair 输出两个方向，仍需本任务的方向竞争和规范化；
- GLiREL 与 REBEL 的公开代码/模型许可含非商业限制，采用前必须做独立许可审查；
- 外部框架的 source ID/description 不等于 exact mention span；
- 调研报告中的 `filecite...`、`cite...` 是不可移植的会话内标记，
  不能作为独立审计链接；
- 报告中的部分命令和 API 是实施草案，不是当前仓库现成可执行入口。

### 18.4 外部方案核验入口

如后续启动可选阶段 X，必须先以当时的官方资料重新核验版本、API 和许可：

- LightRAG 官方仓库：<https://github.com/HKUDS/LightRAG>
- GraphRAG BYOG：<https://microsoft.github.io/graphrag/index/byog/>
- GLiREL 官方仓库：<https://github.com/jackboyla/GLiREL>
- REBEL 官方仓库：<https://github.com/Babelscape/rebel>

## 19. 可选阶段 X：外部关系候选提供器消融

本阶段默认不执行。只有用户在 R0～R9 全部完成后明确授权，且许可证允许时才实施。

### 19.1 前置条件

```text
engineering_complete = true
人工 Gold 与 holdout 已冻结
Native Shadow v2 指标已建立
统一 verifier/normalizer/gate 已通过
新依赖许可证已批准
外部 provider 默认关闭
```

### 19.2 抽象

可定义：

```python
class RelationCandidateProvider(Protocol):
    def propose(
        self,
        *,
        document_revision: str,
        chunk: dict,
        mentions: list[dict],
        schema: dict,
    ) -> list[dict]:
        ...
```

每条外部 hypothesis 必须记录：

```text
provider/name/version
model revision
config hash
raw provider output
document/chunk revision
两端精确 mention occurrence/span
predicate
direction
score
alignment status
```

无法准确对齐两个 occurrence 的输出进入 `UNALIGNED/REJECTED`。

### 19.3 实验顺序

优先做最小候选器，而不是完整 Graph-RAG 集成：

```text
Native Shadow
GLiREL-like external candidate provider
Native ∪ external
必要时再评估 LightRAG re-extract
```

所有 provider：

- 不提供 `related_to` 兜底标签；
- Schema 无对应词映射为 `OTHER_UNSUPPORTED`；
- 两个方向在统一 direction resolver 中竞争；
- 必须经过相同 deterministic validation、verifier、normalization 和 gate；
- 作为可选 extra 或隔离容器固定版本；
- 不改变正式依赖和默认行为。

### 19.4 消融指标

```text
Gold pair/candidate recall
strict final precision/recall/F1
direction accuracy
per-predicate macro-F1
verifier support rate
OTHER_UNSUPPORTED/related_to rate
duplicate/reverse/self-loop
新增 Gold 事实
调用成本、延迟和资源
```

候选数量增加不计为召回提高。

## 20. 建议测试矩阵

文件名按仓库风格调整，覆盖不得减少。

| 测试组 | 重点 |
|---|---|
| pair planner | C(n,2)、顺序稳定、重叠窗口、预算状态、自环 |
| classifier contract | 三类决策、多 hypothesis、方向、完整响应 |
| schema semantics | wildcard、domain/range、symmetric、inverse、review-only |
| evidence verifier | 谓词蕴含、方向、span、否定、假设、失败 |
| normalization | inverse、symmetric、冲突、多谓词 |
| persistence | 幂等、重试、lease、事务、兼容迁移 |
| shadow isolation | 正式表/图/active policy 零写入 |
| quality metrics | multiset、跨 snapshot 对齐、无 Gold 拒算 |
| release gate | 高召回低精度拒绝、未验证/stale 拒绝 |
| freshness | 每个 fingerprint 维度变化触发 stale |
| pilot replay | 176 条完整终态和错误簇报告 |
| entity endpoint characterization | Canada 第三次、lithium/cobalt hard negatives；失败记组合 blocker，不强迫扩域 |

## 21. 推荐启动命令

### 21.1 交互式执行（推荐）

把本文放到仓库根目录，然后执行：

```bash
cd /path/to/Knowledge-Forge

git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short --branch
git diff --name-status
git diff --check

git switch -c fix/shadow-relation-quality-v2

codex --sandbox workspace-write --ask-for-approval on-request
```

若分支已存在或切换失败，不得使用 `-f`；选择唯一分支名或在当前安全分支继续。
不要使用 `--yolo` 或 `danger-full-access`。

进入 Codex 后：

```text
/goal 完成 Shadow 关系发现与可信发布专项；保留高覆盖候选，只有 fresh verifier-supported 关系可进入物化，并用同一人工 Gold 证明最终召回。
```

然后进入 Plan mode：

```text
/plan
```

粘贴：

```text
请完整读取仓库根目录的 Knowledge-Forge-Shadow-Relation-Optimization-Codex-Task.md，
以及所有适用的 AGENTS.md、项目配置、迁移、关系/质量/状态机源码和测试。

先执行任务书 R0，只调查、复现、运行基线测试并生成当前 HEAD 的文件级计划；
R0 期间不得修改文件。历史 cb0d833 仅作问题线索，不得 checkout。

计划必须覆盖 R1～R9 的依赖、允许修改范围、迁移、测试和停止条件。
保留启动前全部用户修改，不 commit、不 push、不开 PR。
```

确认 Codex 的 R0 计划符合本文后，退出 Plan mode并发送：

```text
/plan
```

`/plan` 是切换命令；也可以使用当前 Codex 界面提供的 Plan mode 切换方式。
确认已回到可实施模式后发送：

```text
严格按任务书执行 R0～R9。R1～R7 先写修复前失败的回归测试再实施；
R0、R8、R9 不为形式要求修改算法。运行每阶段门禁并自审；
门禁通过后自动进入下一阶段，不等待逐阶段确认。任一停止条件触发时立即停止并报告。
不执行可选阶段 X，不 commit、不 push、不激活 treatment。
```

### 21.2 非交互式执行

复杂迁移和长阶段更适合交互式执行。仅在隔离、无需审批的环境中才考虑：

```bash
codex exec --sandbox workspace-write - \
  < Knowledge-Forge-Shadow-Relation-Optimization-Codex-Task.md
```

若非交互执行因权限、数据库或用户选择而阻塞，不得放宽权限自动重试。

### 21.3 最终 review

实现与完整测试结束后，在同一 Codex 会话输入：

```text
/review
```

选择 review uncommitted changes，并使用：

```text
审查当前全部未提交修改，并区分启动 Codex 前已有的用户修改。
以 Knowledge-Forge-Shadow-Relation-Optimization-Codex-Task.md 的永久不变量、
R0～R9 门禁和最终自审清单为准。重点查找可导致漏候选、错误方向、错误谓词、
证据错配、重复边、classified 冒充 verified、Shadow 正式写入、stale 绕过和
不兼容迁移的问题。只报告 findings，不修改代码，不 commit，不 push。
```

## 22. 最终判定

本任务的成功标准不是：

```text
176 变成更多
related_to 数量下降但真实关系也被截断
F1 看似不变
测试通过但没有 Gold
```

而是：

```text
Shadow 仍覆盖尽可能完整的关系机会
每个机会有账本
每个正假设有精确证据和方向
每个发布候选经独立 verifier 支持
重复、反向冲突和自环可解释且受控
stale 和未验证结果无法发布
在同一人工 Gold 上，最终严格召回确实高于基线，同时 precision 达到门禁
```

只有同时满足这些条件，才能说 Shadow 从“产生更多关系输出”升级为“发现更多可信关系”。
