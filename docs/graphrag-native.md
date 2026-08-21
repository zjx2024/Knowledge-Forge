# GraphRAG Native 策略

`graphrag_native` 是 Knowledge Forge 产品当前两种项目级构建策略之一。它从当前
active document revision 的已持久化结构块生成 GraphRAG 3.1.1 `standard` 输入，
所有算法事实保留在 run-local Parquet、GraphML 与 LanceDB 中。

它不会调用或写入 Knowledge Forge 策略的实体、关系、别名、pgvector 或 HugeGraph。
native 的 `READY` 只代表 native 的内部 active-run 指针可查询；不会改变项目绑定的
构建策略、不会自动选择该策略，也不会激活 HugeGraph 投影。

## 运行与状态

1. 先完成项目全部文档的预处理。
2. 在“构建方案”中选择 `GraphRAG Native`，执行项目级构建。
3. executor 在新的 staging 目录运行 `graphrag init`、配置/连通性 dry-run 与标准 index。
4. 独立 bundle validator 验证 Parquet、原生关系端点、社区/报告闭包、LanceDB、日志、hash 与凭据扫描后，才可能把 native run 标为 `READY`。

文档上传、删除或任何新的 active revision 都会让 native snapshot 变为 stale。旧 bundle
即使仍存在，也不能回答当前文档状态；新构建失败也不会回退到 Knowledge Forge 策略。

`EMPTY_INPUT`、`NO_GRAPH_UNSUPPORTED`、`FAILED`、`CANCELLED`、`STALE` 与 `SUPERSEDED` 都不可查询。非空输入但没有原生语义关系的 GraphRAG Standard 结果会进入 `NO_GRAPH_UNSUPPORTED`，而不是伪造空图为成功。

`EMPTY_INPUT` 在启动 `graphrag init`、provider probe 或模型调用之前写入控制面；它表示当前 active revisions 没有可安全渲染的结构化输入。相同 source/model/config/prompt/code snapshot 的重建失败不会覆盖旧 READY bundle；但 source snapshot 已变化时，旧 bundle 必须以 stale 拒绝查询。

## 查询与查看

`/graph/view?project=<id>&strategy=graphrag_native` 是应用内只读查看器，不会跳转 Hubble。原生 API 为：

- `GET /api/projects/<id>/graphrag-native/status`
- `POST /api/projects/<id>/graphrag-native/build`
- `POST /api/projects/<id>/graphrag-native/cancel`
- `GET /api/projects/<id>/graphrag-native/graph`
- `GET /api/projects/<id>/graphrag-native/communities`
- `GET /api/projects/<id>/graphrag-native/community-reports`
- `POST /api/projects/<id>/graphrag-native/query`

查询使用单请求子进程中的锁版本 GraphRAG 3.1.1 Python API。`local`、`global`、`drift` 与 `basic` 都只读当前 native bundle；短 ID 必须唯一映射回同一 run 的 native ID 和 source map，否则失败关闭。

其中 `global` 只重新校验并读取 `entities`、`communities`、`community_reports` 三张 Parquet 表，不打开 LanceDB，也不读取 Local/DRIFT 专属的关系或 TextUnit 输入。其他模式只读取各自 GraphRAG 3.1.1 所需的同 run 文件。

## 验证、导出与恢复

```bash
.venv/bin/python scripts/validate_graphrag_native_bundle.py --root '<RUN_DIR>'
.venv/bin/python scripts/check_graphrag_native_freshness.py --project-id '<PROJECT_ID>' --run-id '<RUN_ID>'
.venv/bin/python scripts/export_graphrag_native_bundle.py --project-id '<PROJECT_ID>' --run-id '<RUN_ID>' --output '<NEW_EMPTY_DIR>'
.venv/bin/python scripts/retention_graphrag_native.py --project-id '<PROJECT_ID>'
```

迁移只可前进：发生故障时先备份 PostgreSQL，再修复并重新运行迁移；不要自动 drop 或降级。run 目录不可原地重写，失败/中断使用新的 attempt。默认不删除任何 native bundle；运维清理必须先确认它不是 active、leased 或最后一个 ready run。

`retention_graphrag_native.py` 默认 `dry_run`，只列出 FAILED/CANCELLED/STALE/SUPERSEDED/EMPTY_INPUT/NO_GRAPH_UNSUPPORTED 等非 active、非 lease、非最后 READY 的候选。物理删除需要单独人工授权，并用 `--apply --confirm-run-id <RUN_ID>` 精确确认；它不会删除 PostgreSQL 审计行，也不会触发任何旧策略重建。

真实模型构建、查询、固定 Gold 与重复质量实验是显式高成本步骤，必须单独获得授权。离线测试只能证明接线、隔离和结构契约，不能宣称 precision、recall 或质量提升。
