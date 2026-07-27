# Shadow v2 运行产物审计报告

项目：产业链0724 / project_id `13`<br>
Run ID：`shadow_relation_v2_7dbfa2221b2a86f976c3dc8052e9e7ad5932ccdef2c2e30b4b899863e4cde52f`<br>
Namespace：`entity_shadow_108e5ff3fbbf55ff4e43d582264bccdb323695d76d0dd9e6599aed92bc575b73`<br>
Input fingerprint：`759d7d94cafd80e464c311ea41d6d59eb8b9b86c8cbfeb13da9680bccab8babd`<br>
导出目录：`artifacts/shadow-v2/project-13/2026-07-27-native-v2`

## 文件清单

- `run-summary.json`：运行元数据、计数、状态汇总和一致性检查。
- `response-quality.json`：classifier/verifier 响应 hash、错误和可观察空响应统计。
- `entities.csv`：P2 规范实体及 mention 聚合数。
- `assignments.csv`：每个实体提及到规范实体的归一 assignment。
- `pair-ledger.csv`：完整 pair opportunity ledger，共 4934 行。
- `hypotheses.csv`：classifier 产生的 hypothesis、deterministic validation 与 verifier 终态。
- `verifications.csv`：独立 verifier attempt 明细。
- `normalized-facts-with-evidence.csv`：33 条 normalized fact 及其 evidence、hypothesis、verification provenance。
- `pair-status-summary.csv`、`verification-verdict-summary.csv`：便于快速查看的状态汇总。

## 核心计数

| 产物 | 数量 |
| --- | ---: |
| entity mentions | 337 |
| entity canonicals | 265 |
| entity assignments | 337 |
| pair opportunities | 4934 |
| classifier attempts | 746 |
| hypotheses | 105 |
| validations | 105 |
| verifier attempts | 105 |
| normalized facts | 33 |
| fact evidence rows | 33 |

## Pair 状态

- FILTERED_DISTANCE: 4070
- NO_RELATION: 691
- DEFERRED_BUDGET: 117
- HYPOTHESES: 55
- REJECTED_SELF_LOOP: 1

这里的 `NO_RELATION` 表示 classifier 明确判定该 pair 没有可发布关系，不等同于模型空响应。

## Verifier 终态

- REJECTED: 47
- SUPPORTED: 33
- UNCERTAIN: 25

## 响应质量与空响应

| 阶段 | attempts | error attempts | missing response hash | missing request hash |
| --- | ---: | ---: | ---: | ---: |
| classifier | 746 | 0 | 0 | 0 |
| verifier | 105 | 0 | 0 | 0 |

可观察的模型空/失败响应数量为 0。当前数据库不保存完整原始 LLM 响应正文，因此空响应以 `error_message` 和缺失 `response_hash` 进行审计。

## Normalized fact 对账

| 指标 | 数量 |
| --- | ---: |
| normalized facts | 33 |
| evidence rows | 33 |
| SUPPORTED-backed facts | 33 |
| SUPPORTED-backed ratio | 100.00% |

结论：本次 33 条 normalized fact 均能追溯到 evidence、hypothesis 和 SUPPORTED verifier attempt。

## 注意事项

- 这些产物是 Shadow v2 隔离运行产物，不代表已经激活到正式图。
- 导出文件不包含 PDF 原文、数据库 dump、向量或模型文件。
- 没有人工 Gold 时，这些文件只能证明运行链路、状态契约和可追溯性，不能单独证明 precision/recall 提升。
