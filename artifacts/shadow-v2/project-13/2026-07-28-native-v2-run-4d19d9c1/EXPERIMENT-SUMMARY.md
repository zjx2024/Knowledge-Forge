# Shadow v2 enhanced_shadow 实验总结

本报告对应 Knowledge Forge 项目 `产业链0724`（`project_id=13`）在 2026-07-28 的一次真实入口 `enhanced_shadow` native v2 运行。没有人工 Gold，因此本报告只说明“v2 已按结构契约跑通”和“本轮可追溯产物是什么”，不宣称 precision、recall 或综合质量提高。

## 运行锚点

| 项 | 值 |
| --- | --- |
| 分支 | `fix/shadow-relation-quality-v2` |
| 运行时基线 HEAD | `ae5263bc632f13a76345e2962f955aad5b3f7ab0` |
| run_id | `shadow_relation_v2_4d19d9c1f2cc90fe50cb2d7532a1f8ee51fad8c2a1683c20dada6b234408cb72` |
| input_fingerprint | `9ffcb3aba9b523bd69af73803f39d279b86266d920f8c23ef9329176442a0314` |
| entity snapshot / namespace | `entity_shadow_1b8deffc57a75de0eecbd17bd31da642f246d64c73f6f77efc8bc0ce4ccbb364` |
| source_generation | `12` |
| pipeline_generation | `1` |
| run status | `ready` |
| 开始 / 完成 UTC | `2026-07-28T02:36:12.980507+00:00` / `2026-07-28T02:48:46.792765+00:00` |

## 输入文档与 chunk

本轮实际使用当前 active revision：2 份文档、15 个 chunk。数据库中曾留下历史重建产生的更多 chunk 行，但当前实验按 `documents.active_revision_id = chunks.revision_id` 固定为这 15 个 chunk。

| 文档 | document_id | active_revision_id | parser | chunk 策略 |
| --- | ---: | --- | --- | --- |
| `usgs-cobalt-2025.pdf` | 19 | `revision_6b633dcc26f74e332c37a10e98d5384634d911b000499f5e83561677a5106e08` | `plaintext_pdf` | `general` |
| `usgs-lithium-2025.pdf` | 18 | `revision_6333d5a794c246480a5e13fcbbf750d2cb53c959b2c98f5ad2a8f88e7b8beddf` | `plaintext_pdf` | `general` |

完整输入清单见：

- `active-input-documents.csv`
- `active-input-chunks.csv`

## 真实调用链

实际入口是前端/服务的 `enhanced_shadow` 构建路径，运行链路为：

```text
enhanced_shadow 公共入口
  -> 文档预处理结果复用
  -> P2 entity shadow mention/entity assignment
  -> Shadow v2 unordered occurrence pair planner
  -> shadow_relation_pair_opportunities pair ledger
  -> v2 classifier 生成 0..n hypotheses / NO_RELATION / OTHER_UNSUPPORTED
  -> deterministic validation
  -> independent verifier
  -> direction normalization
  -> shadow_relation_normalized_facts + shadow_relation_fact_evidence
  -> entity_shadow_relation_endpoints 隔离物化
```

旧 P1 管线没有被调用：

- `legacy_pipeline_calls = 0`
- 当前 run 的审计 source label 已修正为：
  - `shadow_relation_pair_opportunities.source_kind = v2_pair_planner`
  - `shadow_relation_hypotheses.source_kind = v2_classifier`
  - `shadow_relation_fact_evidence.source_kind = v2_verifier`

## 关键结果

| 指标 | 数值 |
| --- | ---: |
| pair opportunities 总数 | 3,594 |
| 距离过滤 `FILTERED_DISTANCE` | 2,903 |
| 自环拒绝 `REJECTED_SELF_LOOP` | 1 |
| 预算延后 `DEFERRED_BUDGET` | 66 |
| 进入 classifier 的 pair | 624 |
| classifier attempts | 669 |
| classifier accepted attempts | 624 |
| classifier retryable protocol attempts | 45 |
| accepted empty responses | 0 |
| hypotheses | 87 |
| deterministic validation PASS | 63 |
| deterministic validation REJECTED | 5 |
| deterministic validation REVIEW_ONLY | 19 |
| verifier SUPPORTED | 27 |
| verifier REJECTED | 41 |
| verifier UNCERTAIN | 19 |
| normalized facts | 27 |
| fact evidence rows | 27 |
| 可物化 facts | 27 |
| 每条 fact 有 SUPPORTED verification 的比例 | 100% |
| conflicts | 0 |
| collapse audit rows | 0 |
| proposer persisted hypotheses | 0 |
| proposer failures | 6 |
| proposer skipped by circuit breaker | 13 |
| Stable 正式图新增/修改 | 0 |

这说明“27 条关系”不是候选关系数，而是经过 classifier、确定性校验、独立 verifier 和方向归一之后，最终允许进入隔离 Shadow 图的 supported facts。

## 当前 27 条关系的谓词分布

| predicate | facts |
| --- | ---: |
| `used_in` | 12 |
| `located_in` | 4 |
| `uses` | 4 |
| `part_of` | 2 |
| `produces` | 2 |
| `has_production` | 1 |
| `has_resource` | 1 |
| `is_a` | 1 |

## 与上一轮 33 条结果的差异

上一轮可物化 33 条的 run 是：

`shadow_relation_v2_7dbfa2221b2a86f976c3dc8052e9e7ad5932ccdef2c2e30b4b899863e4cde52f`

| 指标 | 上一轮 33 条 | 本轮 27 条 |
| --- | ---: | ---: |
| pair opportunities | 4,934 | 3,594 |
| classified pairs | 746 | 624 |
| hypotheses | 105 | 87 |
| verifier SUPPORTED | 33 | 27 |
| verifier REJECTED | 47 | 41 |
| verifier UNCERTAIN | 25 | 19 |
| normalized facts | 33 | 27 |

上一轮谓词分布：

| predicate | facts |
| --- | ---: |
| `uses` | 13 |
| `produces` | 9 |
| `part_of` | 5 |
| `belongs_to` | 4 |
| `located_in` | 2 |

本轮数量下降的直接原因是 v2 运行链路更严格：只有 verifier `SUPPORTED` 且具有独立 evidence 的 hypothesis 才能生成可物化 normalized fact；`REJECTED`、`UNCERTAIN`、`NO_RELATION`、`OTHER_UNSUPPORTED` 都不会发布。另一个显著变化是方向归一生效后，部分上一轮看作 `uses` 的关系规范到了 `used_in`。

这不自动等价于“质量变差”。没有人工 Gold 时，只能说本轮结果更保守、审计链更完整；至于少掉的 6 条是真漏召还是上一轮噪声，需要用人工 Gold 或对照审阅确认。

## Stable / enhanced_shadow 状态

| strategy | status | generation | ready_generation | stats |
| --- | --- | ---: | ---: | --- |
| `enhanced_shadow` | `ready` | 20 | 20 | `{"documents":2,"chunks":15,"mentions":318,"entities":254,"relations":27}` |
| `stable` | `ready` | 12 | 12 | `{"documents":2,"chunks":15,"entities":0,"relations":0}` |

本轮 enhanced_shadow 只写隔离 Shadow 产物，没有激活 Stable，正式图新增/修改为 0。

## 产物文件

本目录即本轮实验产物包：

`artifacts/shadow-v2/project-13/2026-07-28-native-v2-run-4d19d9c1/`

核心文件：

- `run-summary.json`：`shadow_relation_v2_runs` 完整运行记录和 metrics。
- `freeze-manifest.json`：run 的 fingerprint 组件和 snapshot 锚点。
- `metric-summary.json`：导出器重算的数量级摘要，`audit_passed=true`。
- `response-quality.json`：模型响应质量摘要，accepted empty response 为 0。
- `pair-ledger.csv`：3,594 条 pair opportunity ledger。
- `pair-status-summary.csv`：pair 规划状态统计。
- `classifier-attempts.jsonl`：669 次 classifier 尝试，含请求/响应审计。
- `hypotheses.csv`：87 条 hypothesis。
- `verifications.csv`：87 条 verifier attempt。
- `verification-verdict-summary.csv`：verifier 终态统计。
- `normalized-facts-with-evidence.csv`：27 条 normalized fact 及 evidence_count。
- `fact-evidence.jsonl`：27 条 fact evidence，均链接 SUPPORTED verification。
- `audit-report.md`：导出器生成的审计摘要。
- `active-input-documents.csv`：本轮输入文档 revision。
- `active-input-chunks.csv`：本轮 15 个 active chunk hash。

说明：`model-calls.jsonl`、`proposal-attempts.jsonl`、`proposals.csv`、`structure-contexts.jsonl` 等扩展审计文件为空，是因为当前数据库对这次 run 的可审计链主要落在 pair/classifier/verifier/fact evidence 表；这不影响 27 条 fact 的 SUPPORTED evidence 对账。

## 验证命令与结果

导出器结果：

```text
metric_summary.audit_passed = true
supported_verification_for_fact_evidence_ratio = 1.0
response_quality.accepted_empty_responses = 0
```

代码与测试：

```bash
.venv/bin/python -m py_compile knowledge_forge/database.py knowledge_forge/migrations.py knowledge_forge/relation_shadow_v2.py knowledge_forge/relation_shadow_runtime.py knowledge_forge/config.py knowledge_forge/web.py server.py
.venv/bin/python -m pytest --strict-markers -q tests/test_shadow_relation_runtime_wiring.py tests/test_shadow_relation_classifier_v2.py tests/test_shadow_relation_retry_v2.py tests/test_shadow_relation_persistence_v2.py
```

结果：

```text
50 passed, 2 skipped in 0.42s
```

产物导出测试：

```bash
.venv/bin/python -m pytest --strict-markers -q tests/test_shadow_v2_artifact_exporter.py tests/test_shadow_v2_artifact_contract.py
```

结果：

```text
3 passed, 1 skipped in 0.28s
```
