# Stable 基线运行产物审计报告

项目：产业链0724 / project_id `13`<br>
策略：`stable`<br>
导出目录：`artifacts/stable-baseline/project-13/2026-07-27-stable`

## 文件清单

- `run-summary.json`：stable graph、document build、计数和状态汇总。
- `document-builds.csv`：两份文档的 stable 构建记录。
- `entities.csv`：stable 规范实体及 mention 聚合数。
- `mentions.csv`：stable 实体提及及其 canonical 映射。
- `relation-assertions.csv`：stable PostgreSQL 关系断言，共 10 条。
- `relation-candidates.csv`：stable 关系候选，共 10 条。
- `entity-type-summary.csv`、`relation-status-summary.csv`、`candidate-status-summary.csv`：快速状态汇总。

## 核心计数

| 指标 | 数量 |
| --- | ---: |
| graph stats relations | 0 |
| canonical entities | 169 |
| entity mentions | 173 |
| PostgreSQL relation assertions | 10 |
| relation candidates | 10 |
| verified relation assertions | 0 |
| failed relation assertions | 10 |
| assertions with graph edge | 0 |
| assertions missing evidence text | 6 |

## 为什么页面显示 stable=0

stable 的项目图状态中 `stats.relations=0`，并且这 10 条 PostgreSQL `relation_assertions` 全部是 `verification_failed`，`graph_edge_id` 为空。因此它们没有进入正式图可见关系。

这和 enhanced Shadow v2 的 33 条 normalized facts 不同：enhanced 的 33 条均有 SUPPORTED verifier attempt 和 fact evidence。

## relation_assertions 状态

- verification_failed / part_of: 3 条，graph_edges=0
- verification_failed / <empty>: 2 条，graph_edges=0
- verification_failed / located_in: 2 条，graph_edges=0
- verification_failed / produces: 2 条，graph_edges=0
- verification_failed / related_to: 1 条，graph_edges=0

## relation_candidates 状态

- verification_failed / part_of: 3 条
- verification_failed / <empty>: 2 条
- verification_failed / located_in: 2 条
- verification_failed / produces: 2 条
- verification_failed / related_to: 1 条

## 注意事项

- stable 不运行 Shadow v2，因此没有 pair ledger、hypothesis ledger 或 independent verifier ledger。
- 本包导出的 10 条关系记录是 stable PostgreSQL 里的失败断言/候选，用来解释 `stable=0` 的来源。
- 没有人工 Gold 时，不能用这些产物声明 baseline 或 enhanced 的 precision/recall。
