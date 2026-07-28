# Project 14 enhanced_shadow Shadow v2 run summary and manual audit

运行对象：`产业链0728`

运行入口：正式 `enhanced_shadow` ingestion 入口

运行时间：2026-07-28 16:30:20 CST 至 2026-07-28 17:02:21 CST

Shadow v2 run：`shadow_relation_v2_ad9020b5766fd088ff66e8841765b1e8b86cff2670d3ec03c3bbb79e5a0fc326`

Entity namespace：`entity_shadow_113e7ea7c7a3b05353eabb7002e9df6f7abe3167609a1d8aeb7c5f0274c08f86`

文档：

- `usgs-lithium-2025.pdf`
- `usgs-cobalt-2025.pdf`

## 结论

本轮 enhanced_shadow 的工程链路已经稳定跑通：真实入口进入 Shadow v2，旧 P1 管线调用次数为 0，pair ledger、classifier、verifier、normalization 和隔离图物化均有过程产物；37 条 normalized facts 的 41 条 evidence 均可追溯到 `SUPPORTED` verifier，比例为 100%。

但按人工核验口径，本轮实体、关系、归一质量还不适合作为稳定版质量基线。主要问题不是系统失败，而是语义质量：表头/文档元信息进入图谱、列表项误连、谓词方向错误、`produces`/`used_in` 角色约束不足，以及化学品实体归一误合并。

## 过程产物

本目录是本次 run 的完整过程文件 bundle：

- `run-summary.json`：Shadow v2 run 元数据、fingerprint 与运行 metrics。
- `metric-summary.json`：导出器重算后的过程计数和审计通过状态。
- `audit-report.md`：bundle 结构审计报告。
- `freeze-manifest.json`：输入 fingerprint、schema/policy/model/prompt 等冻结信息。
- `structure-contexts.jsonl`、`source-maps.jsonl`：Shadow v2 使用的 source contexts 与 source map。
- `mentions.csv`、`entities.csv`、`assignments.csv`、`mention-appearances.csv`、`occurrences.csv`：实体提及、canonical 与归一过程文件。
- `pair-ledger.csv`、`pair-contexts.csv`、`pair-status-summary.csv`：pair planner / ledger 过程文件。
- `classifier-attempts.jsonl`、`hypotheses.csv`：classifier 尝试和 hypothesis 过程文件。
- `proposal-attempts.jsonl`、`proposals.csv`：direct proposer 尝试文件。
- `verifications.csv`、`verification-verdict-summary.csv`：独立 verifier 过程文件。
- `normalized-facts-with-evidence.csv`、`fact-evidence.jsonl`：normalized facts 与精确 evidence/provenance。
- `collapse-audit.jsonl`、`conflicts.csv`：事实归一去重与冲突审计。
- `manual-verification.csv`：人工核验逐条事实判定。
- `manual-verification-summary.json`：人工核验统计摘要。

注意：当前实现中 `shadow_relation_proposals` 表为空，因此 `proposals.csv` 为空；direct proposer 产生的 141 条候选已经进入 `hypotheses.csv`，其 `source_kind` 为 `v2_DIRECT_PROPOSER`。最终 facts 中有 8 条 evidence 来自 direct proposer。

## 运行指标

| 指标 | 数量 |
|---|---:|
| source contexts | 15 |
| pair opportunities | 3,765 |
| 实际进入分类的 pairs | 717 |
| classifier attempts | 733 |
| classifier accepted decisions | 717 |
| classifier retryable protocol attempts | 16 |
| classified hypotheses | 234 |
| direct proposer hypotheses | 141 |
| classifier hypotheses | 93 |
| verification attempts | 234 |
| verifier `SUPPORTED` | 41 |
| verifier `REJECTED` | 176 |
| verifier `UNCERTAIN` | 17 |
| normalized facts | 37 |
| fact evidence rows | 41 |
| graph materialization eligible facts | 36 |
| graph endpoints | 36 |
| legacy P1 pipeline calls | 0 |
| supported verification for fact evidence | 100% |

运行稳定性观察：

- 本轮状态为 `ready`。
- direct proposer 没有熔断；40 次 proposal attempts 中 39 次 accepted，1 次 retryable failed。
- classifier 有 16 次 retryable protocol failure，但最终没有 terminal failure。
- 所有发布 evidence 均来自独立 verifier 的 `SUPPORTED` verdict。
- 没有 `REJECTED`、`UNCERTAIN`、`RETRYABLE_FAILED`、`NO_RELATION` 或 `OTHER_UNSUPPORTED` 被发布为 normalized fact。

## 关系分布

| predicate | facts | evidence rows |
|---|---:|---:|
| `used_in` | 14 | 14 |
| `located_in` | 5 | 6 |
| `is_a` | 4 | 6 |
| `produces` | 4 | 5 |
| `part_of` | 3 | 3 |
| `uses` | 3 | 3 |
| `belongs_to` | 1 | 1 |
| `causes` | 1 | 1 |
| `has_production` | 1 | 1 |
| `substitutes_for` | 1 | 1 |

## 实体与归一观察

实体产物：

- entity mentions：324
- canonical entities：260
- relation endpoints：36

类型分布：

| primary_type | canonical entities | mention sources |
|---|---:|---:|
| Product | 100 | 112 |
| Location | 86 | 124 |
| GenericEntity | 34 | 39 |
| Organization | 18 | 21 |
| Event | 11 | 13 |
| Technology | 10 | 13 |
| Person | 1 | 2 |

归一总体偏保守，这是好事；但存在一处严重误合并：

- `lithium carbonate` 与 `lithium chloride` 被合并到同一 canonical。

这说明 Product/chemical/mineral 类实体不能只依赖表面相似或宽松 lexical merge，需要更严格的化学品等价规则。

实体噪声也比较明显，以下类型不应直接参与领域知识图谱关系：

- 电话号码、联系人信息；
- `Item Number Normal Trade Relations`、`Government Stockpile`、`World total` 等表头/栏目；
- `Mineral Commodity Summaries`、`January 2025` 等文档元信息。

## 人工核验结果

人工核验范围：本轮 37 条 normalized facts。核验依据是系统导出的 evidence span 与对应原文上下文。

这是“已发布 fact 的人工准确性核验”，不是 precision/recall 的完整 Gold 评测；没有覆盖系统漏召回的事实。

| 判定 | 数量 | 比例 |
|---|---:|---:|
| `correct` | 16 | 43.2% |
| `needs_correction` | 13 | 35.1% |
| `reject` | 8 | 21.6% |

解释：

- `correct`：实体、谓词、方向和证据基本成立，可保留。
- `needs_correction`：原文有相关信息，但谓词、方向、端点或实体粒度需要修正。
- `reject`：应拒绝，不应进入图谱。

严格可直接保留率约 43.2%；若把可修正事实也算作“有信息基础”，则约 78.4% 的发布事实有一定原文基础。

## 主要正确关系

可直接保留的代表性关系：

- `Eagle Mine located_in Michigan`
- `Eagle Mine is_a nickel-copper mine`
- `Cobalt cathode is_a refined cobalt metal`
- `ceramic-metallic composites is_a cermets`
- `cobalt-iron-copper used_in diamond tools`
- `iron-cobalt-nickel used_in cutting and wear-resistant materials`
- `lead used_in paints`
- `nickel-base alloys used_in jet engines`
- `superalloys used_in aircraft gas turbine engines`
- `zinc used_in primary batteries`
- `lithium-ion battery industry uses COBALT`

这些关系可以从 evidence span 中直接读出，方向基本合理。

## 主要错误类型

### 1. 文档元信息和表头入图

应拒绝：

- `Tariff belongs_to Item Number Normal Trade Relations`
- `Mineral Commodity Summaries part_of January 2025`

这些属于文档/表格元信息，不是领域实体关系。

### 2. 列表项误连

应拒绝：

- `Argentina located_in Zimbabwe`

原文是国家和矿业活动列表，系统把同一句列表里的两个国家误连成地理包含关系。

### 3. 谓词方向错误

需要修正：

- `aircraft gas turbine engines used_in superalloys`

应为：

- `superalloys used_in aircraft gas turbine engines`

类似问题也出现在 `produces`：

- `refined cobalt metal produces electrolytic process`

应为：

- `refined cobalt metal produced_by electrolytic process`

或在当前 schema 里保守拒绝。

### 4. `produces` 语义过宽

需要修正或拒绝：

- `Cobalt cathode produces electrolytic process`
- `Cobalt cathode produces refined cobalt metal`
- `Automobile companies produces battery materials`

这些文本分别表达“生产工艺”“定义/is-a”“合作供应来源”，不是标准的 producer/product 关系。

### 5. 分号/并列短语跨项误连

应拒绝：

- `stearates substitutes_for glass manufacture`
- `paints used_in unsaturated polyester resins`
- `strontium ferrites used_in lead`

这些错误来自替代品/应用列表中跨分号或跨并列项连接。

### 6. 缩写被建成 `is_a`

需要修正：

- `electric vehicles is_a EVs`

这里应进入实体归一或 `same_as`/alias 机制，而不是领域图谱里的 `is_a`。

## 对质量优化的建议

下一轮建议不要先追求更多关系，而是优先做质量门禁：

1. 为 `used_in`、`produces`、`located_in`、`part_of`、`substitutes_for` 加强 subject/object role 校验。
2. 把表头、页眉、栏目名、日期、电话、联系人、文档标题类实体降级为 metadata/noise，不参与关系。
3. 对分号列表、国家列表、替代品列表建立结构边界，禁止跨项误连。
4. Product/chemical/mineral 归一改为更严格策略；类似 `lithium carbonate` 与 `lithium chloride` 不得因表面相近自动合并。
5. verifier 需要显式判断“该谓词 + 该方向 + 该端点组合”是否被文本支持，而不是只判断两个实体在证据中共同出现。
6. `produced by`、`used as`、`substitutes for ... in ...` 这类表达应进入 schema/policy 的方向说明和 negative examples，不写成生产代码中的特化分支。

## 复核命令

```bash
.venv/bin/python scripts/validate_shadow_v2_bundle.py \
  artifacts/shadow-v2/project-14/2026-07-28-native-v2-run-ad9020b5
```

结果：

```text
ok: true
failures: []
facts: 37
fact_evidence: 41
eligible_facts: 36
```
