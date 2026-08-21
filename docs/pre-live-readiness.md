# 上线前检查与恢复门禁

本文是当前 Knowledge Forge 服务的运维检查清单。产品提供两种相互隔离的项目级构建
策略：`knowledge_forge` 与 `graphrag_native`。历史实验中的 control/shadow/canary 名称
不再作为当前产品策略或发布流程。

## 1. 数据与版本边界

- PostgreSQL 保存项目、文档 revision、结构块、知识块、向量、逻辑 Schema、规范实体、
  关系断言、来源证据、构建代次与任务状态。
- HugeGraph 的 `knowledge_forge` 物理图只承载 Knowledge Forge 策略发布后的可导航投影。
- GraphRAG Native 的算法事实保存在不可变的 run-local Parquet、GraphML 与 LanceDB
  bundle；PostgreSQL 只保存控制面元数据。
- 文档重入库追加 revision。失败 revision 不得覆盖最后一个成功的 active revision。
- 删除文档采用软删除；仍有其他文档支持的事实不得随单个文档删除。

## 2. 上线前只读预检

1. 确认 Git 工作树、目标 commit、配置文件和部署包完全对应。
2. 备份 PostgreSQL，记录 HugeGraph graph、服务版本和当前项目构建代次。
3. 检查 PostgreSQL、pgvector、Ollama/Embedding、reranker、HugeGraph 的健康与认证。
4. 检查所有运行中的 `graphrag_native_build`。只要存在运行中的 Native 构建，就不得
   重启 `knowledge-forge.service`；应等待完成，或在用户明确授权后安全取消。
5. 查看迁移计划和状态；已应用迁移不可原地修改，只能追加 forward-fix migration。

```bash
.venv/bin/python scripts/manage_migrations.py plan
.venv/bin/python scripts/manage_migrations.py status
.venv/bin/python scripts/verify_postgres_backup_restore.py \
  --docker-container knowledge-forge-postgres
```

## 3. 代码与数据库验证

最低门禁：

```bash
.venv/bin/python -m py_compile knowledge_forge/*.py server.py
.venv/bin/python -m pytest -q
```

涉及 PostgreSQL schema、事务、任务租约或持久化 API 时，再运行一次性数据库验证：

```bash
.venv/bin/python scripts/run_disposable_postgres_tests.py \
  -m integration -q -ra --strict-markers
```

测试数据库、schema、role 和 sentinel 必须随机隔离，测试结束只清理本轮创建的资源。
不得把 fixture 结果描述为真实模型质量结论。

## 4. Knowledge Forge 构建门禁

- 构建输入必须绑定 active document revision、结构块/知识块哈希、逻辑 Schema 版本和
  模型配置。
- 实体与关系都必须保留 `project_id`、文档、知识块和页码/结构位置来源。
- 实体归一保持保守；歧义候选分开。关系只有通过端点、方向、谓词、原子事实和来源证据
  校验后才能发布。
- 图发布采用新的 generation 与 lease。预期清单、写入方清单和 HugeGraph 实际清单完全
  一致，且不存在未完成 outbox/work item 后，才能 CAS 切换为 ready。
- 外部 Ollama、Embedding、reranker 和 HugeGraph 调用期间不得长期持有数据库事务。

## 5. GraphRAG Native 构建门禁

- Native 只读取当前结构块并生成独立 bundle，不得调用 Knowledge Forge 的 Chunk、
  pgvector、规范实体、关系管线或 HugeGraph。
- run 目录不可原地重写。失败、取消或恢复都创建新 attempt。
- bundle validator 必须检查 Parquet 闭包、关系端点、社区/报告、LanceDB、来源映射、哈希
  与凭据泄露，验证通过且来源仍 fresh 后才可进入 `READY`。
- `EMPTY_INPUT`、`NO_GRAPH_UNSUPPORTED`、`FAILED`、`CANCELLED`、`STALE`、
  `SUPERSEDED` 均不可查询；不得静默回退到另一策略或旧来源版本。

```bash
.venv/bin/python scripts/validate_graphrag_native_bundle.py --root '<RUN_DIR>'
.venv/bin/python scripts/check_graphrag_native_freshness.py \
  --project-id '<PROJECT_ID>' --run-id '<RUN_ID>'
```

## 6. 真实服务验证

单元测试通过后，针对隔离项目走一次真实公共 HTTP/任务入口，并等待任务进入终态；检查
任务记录、持久化结果、来源证据与新鲜度。模型工作流必须使用部署配置中的真实模型服务，
不能只用 mock 替代。

服务基础检查：

```bash
curl --fail http://127.0.0.1:8011/health
curl --fail http://127.0.0.1:8080/graphs
```

只读检查不等于构建验证。修改上传、预处理、构建、查询或模型调用后，必须额外执行对应的
真实入口。出现 stale lease、迁移缺口、模型响应不完整、任务假成功或来源丢失时，不得发布。

## 7. 网络与凭据

- 未认证的 Web、PostgreSQL、HugeGraph、Ollama、Embedding 或 reranker 服务默认只绑定
  loopback；暴露到局域网需要认证、访问控制和显式运维配置。
- 不提交 `.env`、上传文档、向量、模型、数据库 dump、运行日志、Gold 正文或生成 artifact。
- 运行配置变更同步更新 `.env.example`、Docker 文件与 README。

## 8. 故障恢复

- 数据库迁移只前进修复，不自动 drop、降级或覆盖已应用 migration。
- Knowledge Forge 新图代次失败时保留上一份已验证代次，但不得把旧代次冒充当前来源已更新。
- GraphRAG Native 新 run 失败时不激活半成品；来源已变化时，旧 READY bundle 必须以 stale
  拒绝查询。
- 物理删除 bundle、项目数据或图数据需要精确目标与用户授权；优先使用软删除和可恢复操作。
- 修复完成后重新执行本清单，记录 commit、配置哈希、迁移状态、真实任务 ID 和验证结果。
