# 上线前隔离验证、恢复与发布门禁

本文记录 Knowledge Forge 当前的 revision、shadow、图代次、迁移和隔离验证操作。
默认生产策略仍为 `control`。没有独立人工 Gold、冻结 holdout 和 canary 证据时，
P1/P2 treatment 只能运行在 shadow namespace，不能发布到正式检索或 HugeGraph。

## 1. 文档 revision 与恢复

同一文档每次重入库都会追加 `document_content_revisions`，状态依次经过
`pending -> processing -> active/failed/superseded/cancelled`。`documents.active_revision_id`
只指向最后一次成功激活的 revision；失败或乱序完成不能覆盖仍可读取的旧 active revision。
Chunk、实体提及、关系、审计和 snapshot 输入均绑定 `revision_id`，历史 revision 不通过
外键级联硬删除。

- 重入库：创建新 revision 和 lease；成功后以 CAS 激活并 supersede 旧 revision。
- 失败恢复：旧 active revision 保持可读；新 revision 可按持久化 work item 重试。
- 删除/恢复：文档软删除不会改写历史审计；恢复后再次重入库会创建新 revision。
- 审计清理：受 `AUDIT_REDACTION_ENABLED`、`AUDIT_RETENTION_DAYS`、
  `AUDIT_MAX_ROWS` 和 `AUDIT_CLEANUP_ENABLED` 控制，不删除仍被 snapshot 引用的数据。

业务读取使用 active revision；审计、追责和离线评估必须显式查询历史 revision，不应把两者
混为同一个结果集合。

## 2. Policy snapshot 与 shadow

`relation_only` 从冻结的原始 Chunk 独立运行关系 treatment；`end_to_end` 从冻结 Chunk
独立运行实体与关系 treatment。snapshot manifest 固定项目、文档 revision、Chunk hash、
模型、策略、Git tree、Gold manifest 和阈值，work item 通过短 lease 支持恢复和取消，artifact
写入后不可变。

```bash
.venv/bin/python -m knowledge_forge.policy_snapshot build \
  --project-id <PROJECT_ID> --mode relation_only
.venv/bin/python -m knowledge_forge.policy_snapshot build \
  --project-id <PROJECT_ID> --mode end_to_end
.venv/bin/python -m knowledge_forge.policy_snapshot status --snapshot-id <SNAPSHOT_ID>
.venv/bin/python -m knowledge_forge.policy_snapshot resume --snapshot-id <SNAPSHOT_ID>
.venv/bin/python -m knowledge_forge.policy_snapshot cancel --snapshot-id <SNAPSHOT_ID>
```

队列状态为 `pending/leased/retryable_failed/completed/terminal_failed/cancelled`。重试受最大
attempt、退避、lease expiry 和队列上限约束；runner 结束时 deferred 必须归零或进入明确的
terminal 状态。外部 Ollama、Embedding、reranker 或 HugeGraph 调用期间不得持有数据库事务。

## 3. Ollama 请求契约

项目上下文预算、最大输出和安全余量相互独立：

- 应用通过 `/api/show` 读取模型声明上下文，通过 `/api/ps` 读取当前 runner 上下文；
- 有效上下文取模型声明、runner 和项目预算的最小值；
- `max_output_tokens` 只映射到 `options.num_predict`；
- 应用默认不发送 `options.num_ctx`，避免修改或重载全局 Ollama runner；
- 输入估算 + 输出预留 + 安全余量超过有效上下文时 fail closed；
- `done_reason=length`、不完整响应或 token 统计越界不进入后续 JSON 管线。

## 4. 实体、关系与回滚

实体 exact 命中不调用 Embedding、reranker 或 LLM；非精确 mention 使用 alias/trigram 与
pgvector Top-K 有界召回，再进入 reranker/LLM 裁决。关系分类按紧凑批次运行，同一 Chunk
context 在单次 Prompt 中只出现一次，超出预算的候选进入 deferred，禁止静默截断。

P2 的 merge、split、undo 都追加审计事件，不改写原始 mention 和 relation assertion。
正式关系只接受当前允许 policy/version 且可发布的验证状态。shadow 数据不写正式 canonical、
assertion、vector、search 或 graph namespace；回滚通过恢复上一 policy snapshot 和重建图代次完成。

## 5. Gold、完整分母与 release gate

评估必须使用冻结 manifest、完整 prediction/Gold universe 和不可变 artifact hash。missed Gold、
spurious prediction、未完成 work item 均进入指标分母；pairwise/B3、严格关系/evidence、校准、
成本和 control/treatment 差异不能只在交集上计算。

```bash
.venv/bin/python benchmarks/evaluate_kg_gold.py \
  --snapshot-id <SNAPSHOT_ID> \
  --gold-manifest <HUMAN_GOLD_MANIFEST.json> \
  --output artifacts/evaluation.json
.venv/bin/python -m knowledge_forge.release_gate \
  --evaluation-run-id <EVALUATION_RUN_ID>
```

仓库 fixture 和模型自产标签只用于工程验证，不能生成 `production_approved`。发布路径为
`control -> shadow -> canary -> limited -> expanded`，任一质量、成本、完整性或人工审批门禁
失败时保持 control。

## 6. HugeGraph namespace 与状态

业务写入绑定物理 graph、随机/持久 namespace、project、generation、policy 和 graph lease。
状态为 `stale -> rebuilding -> ready`，失败进入 `failed`；只有 expected、writer-reported 和
actual 三份内容 manifest 在同一 generation/lease 下完全一致，且无待处理 outbox/work item，
才能 CAS 为 `ready`。数量相同但端点、方向、predicate、label 或 generation 不同仍失败。

任何测试 mutation 必须使用非默认空图、UUID namespace、一次性 sentinel 和双开关。第一步
永远只读：

```bash
.venv/bin/python scripts/hugegraph_test_guard.py preflight \
  --host <TEST_HOST> --port 8080 --graph <NON_DEFAULT_EMPTY_GRAPH> \
  --namespace <UUID>
```

只读 integration 也必须保持零 mutation：

```bash
RUN_HUGEGRAPH_INTEGRATION=1 \
.venv/bin/python -m pytest -q -ra --strict-markers \
  -m "hugegraph_integration and not destructive"
```

只有物理隔离得到证明后才可 arm；arm、执行和 cleanup 必须使用同一个 run、namespace 和
短期 sentinel：

```bash
RUN_HUGEGRAPH_INTEGRATION=1 RUN_HUGEGRAPH_DESTRUCTIVE_TESTS=1 \
.venv/bin/python scripts/hugegraph_test_guard.py arm \
  --host <TEST_HOST> --port 8080 --graph <NON_DEFAULT_EMPTY_GRAPH> \
  --namespace <UUID> --run-id <RANDOM_RUN_ID>

RUN_HUGEGRAPH_INTEGRATION=1 RUN_HUGEGRAPH_DESTRUCTIVE_TESTS=1 \
HUGEGRAPH_TEST_GRAPH=<NON_DEFAULT_EMPTY_GRAPH> GRAPH_NAMESPACE_ID=<UUID> \
HUGEGRAPH_TEST_SENTINEL=<ONE_TIME_SENTINEL> \
.venv/bin/python -m pytest -q -ra --strict-markers \
  -m "hugegraph_integration and destructive"

RUN_HUGEGRAPH_INTEGRATION=1 RUN_HUGEGRAPH_DESTRUCTIVE_TESTS=1 \
.venv/bin/python scripts/hugegraph_test_guard.py cleanup \
  --host <TEST_HOST> --port 8080 --graph <NON_DEFAULT_EMPTY_GRAPH> \
  --namespace <UUID> --run-id <RANDOM_RUN_ID> \
  --sentinel <ONE_TIME_SENTINEL>
```

默认 `knowledge_forge` 图、身份未知、与 production 相同、sentinel 过期/不匹配或任一开关缺失
时必须零 mutation 拒绝。共享物理图不得执行 schema drop、全图 purge 或 destructive failure
injection。

## 7. 迁移、备份与显式清理

```bash
.venv/bin/python scripts/manage_migrations.py plan
.venv/bin/python scripts/manage_migrations.py status
.venv/bin/python scripts/manage_migrations.py apply
.venv/bin/python scripts/verify_postgres_backup_restore.py \
  --docker-container knowledge-forge-postgres
```

迁移使用 checksum、advisory lock、独立状态和可恢复 backfill。已应用迁移禁止原地修改，使用
forward-fix。服务启动只应用快速安全 migration；重 backfill/HNSW 由运维显式执行。旧列清理
默认只展示计划，执行必须同时提供可追溯备份确认和显式 destructive 开关，详见
[`migrations-and-release-safety.md`](migrations-and-release-safety.md)。

## 8. 隔离测试与性能门禁

真实 PostgreSQL 测试会创建随机 database、随机 schema、受限 role 和 admin-owned sentinel，
验证 PostgreSQL 16/pgvector、FK、迁移、事务、并发、重入库和 EXPLAIN，最后仅清理本次随机
资源：

```bash
.venv/bin/python scripts/run_disposable_postgres_tests.py \
  -m integration -q -ra --strict-markers
```

统一 E2E 第一步始终是零写预检：

```bash
.venv/bin/python scripts/run_isolated_e2e.py --preflight-only
```

只有清单证明 PostgreSQL、Ollama/Embedding/reranker、HugeGraph、API、namespace、sentinel、
双开关、Git tree 和配置 hash 均为隔离资源时才返回短期 arm token；环境发生变化 token 立即
失效。条件不完整时退出码为 2，禁止执行。

性能硬门禁以操作数而非单机耗时为主：exact hit 零模型调用、reranker pairs 不超过
`mentions * K`、无项目全扫描、relation vector 在 LIMIT 前过滤、10 文档稳定批次最多一次
完整 rebuild、队列/lease/outbox 有界且最终无未声明积压。质量门禁优先，不能通过丢候选、
放宽 evidence 或扩大误合并换取性能。

## 9. 网络、凭据与当前未验证项

裸机和宿主 published port 默认绑定 `127.0.0.1`。无认证服务禁止暴露到共享 UAT、互联网或
不可信网络；production/shared 绑定非 loopback 需要显式危险开关、认证和运维批准。PostgreSQL、
HugeGraph 和 arm secret 不提供弱默认值，不得提交 `.env`、语料、dump、日志、Gold 正文或任务文档。

本轮本地真实验证覆盖 disposable PostgreSQL 16 + pgvector。若没有单独 Ollama endpoint、独立
非默认空 HugeGraph 图或独立人工 Gold，则对应 live smoke/destructive graph/production gate
保持未验证和关闭；fake/fixture 测试不能替代这些结果。
