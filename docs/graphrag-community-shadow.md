# GraphRAG 社区检测 Shadow

Knowledge Forge 已接入 GraphRAG 3.1.1 的 BYOG 社区检测边界。该模块只读取
PostgreSQL 当前正式 control 图，执行 `create_communities`，并把运行、社区层级、
成员和聚合边溯源保存回 PostgreSQL。结果保持 shadow，不参与现有检索、问答或
HugeGraph。

## 正式图来源

输入关系严格沿用产品查询的 fail-closed 语义：

- 只读取文档 `active_revision_id` 对应的当前 revision；
- 使用 `canonical_relation_projections` 投影后的规范实体端点；
- 排除软删除、不可发布和非当前验证策略的关系；
- 自动验证关系必须属于当前 verifier policy，人工验证关系独立允许；
- 不读取 `entity_shadow_*` 或 `relation_shadow_*` treatment 数据；
- 运行开始和提交前分别校验 graph generation、policy 和完整输入指纹。

当前 active revision 中仍有有效 mention 支撑的规范实体都会进入 BYOG 节点，
不要求该实体已经出现在可发布关系中。实体 `title` 使用项目内唯一的 `entity_key`，
显示名称、类型和描述放入 `description`。Chunk 来源使用稳定的
`chunk:<revision_id>:<chunk_key>` 标识。没有任何边的实体会形成稳定的单点社区，
不会因为关系召回不足而从社区快照中消失。

## 边和权重

首版权重策略固定为 `publishable_assertion_count_v1`：

- 按无序规范实体端点聚合反向边和重复边；
- 每条可发布断言贡献 `1.0` 权重；
- confidence 不作为结构强度；
- 保存 assertion、Chunk、关系类型和 text unit 溯源；
- 自环排除并计数，悬空端点和非法权重失败关闭；
- `use_lcc=false`，所有非连通分量都参与 Leiden；
- 空关系图直接记为 `completed_empty`，不调用 GraphRAG。

第一阶段始终执行完整快照检测，不使用 `graphrag update`。

## CLI

应用迁移后，显式执行：

```bash
.venv/bin/python -m knowledge_forge.graphrag_community detect \
  --project-id 1 --seed 42 --max-cluster-size 10 --timeout 300
```

只读查看运行状态和结果：

```bash
.venv/bin/python -m knowledge_forge.graphrag_community status --run-id <RUN_ID>
.venv/bin/python -m knowledge_forge.graphrag_community show --run-id <RUN_ID>
```

`detect` 是唯一执行入口，不会由服务启动、文档入库或定时任务自动触发。

## 持久化状态

- `graphrag_community_runs`：输入指纹、generation、policy、GraphRAG/适配器版本、
  配置、指标和 `running/completed/completed_empty/failed/superseded` 生命周期；
- `graphrag_communities`：层级、原 GraphRAG 标识和稳定社区 ID；
- `graphrag_community_members`：社区到规范实体的成员关系；
- `graphrag_community_edges`：本次 BYOG 聚合边及完整来源。

稳定社区 ID 由项目、输入指纹、层级和排序后的规范实体 ID 计算，不依赖 GraphRAG
随机 UUID 或 Leiden 整数。检测期间正式图发生变化时，运行进入 `superseded`；失败
不会产生伪 `completed`。

## Community Reports shadow

报告阶段显式调用 GraphRAG 3.1.1 的 `create_community_reports` 工作流。报告运行绑定：

- project、graph generation 和 community detection run；
- community input、prompt 和模型配置 hash；
- GraphRAG 与模型版本、attempt、时延、usage 和错误类别；
- 同一快照下的 community、entity 和 relation citation。

报告按 community 幂等执行。输入未变化且已有成功报告时不会重复调用模型；超时和
连接错误进入 `retryable_failed`，可继续相同运行；格式、引用等不可恢复错误进入
`terminal_failed`。只有全部 eligible community 成功时，整个报告运行才原子切换为
active。失败或部分完成的运行不会替换上一个 active 运行。空图产生 0-report 的成功
运行，且模型调用数为 0。

GraphRAG 原生报告不携带 Knowledge Forge 的规范实体和关系 ID。适配器会把本次
community 的精确成员和聚合边 ID 附加到 finding，数据库再次校验这些 ID 属于同一
project、detection snapshot 和 community。关系可通过聚合边 provenance 追溯到
active revision 的 assertion 和 Chunk；实体可通过规范实体对应的 active mention
追溯。这里提供的是可解析的快照级溯源，尚不能替代人工 Gold 对每个自然语言 claim
做事实支持判定。

显式执行和查看：

```bash
.venv/bin/python -m knowledge_forge.community_reports run <DETECTION_RUN_ID> \
  --graphrag-root <GRAPHRAG_PROJECT_DIR>
.venv/bin/python -m knowledge_forge.community_reports show <REPORT_RUN_ID>
```

## Global Search opt-in shadow

Global Search 使用 GraphRAG 3.1.1 的 `graphrag.api.query.global_search`，只读取当前
project 下 active、完整且仍对应当前 graph generation 的报告快照。调用必须显式
opt-in；默认产品检索、问答和 HugeGraph 路径没有改变。缺少 active 报告、报告为空、
快照 stale、跨项目引用或 map/reduce 失败时均 fail closed，不会把部分结果标为成功。

每次查询保存 query/report/prompt/model config hash、GraphRAG 版本、被选报告、引用、
map/reduce 输出、usage、时延和错误类别。默认 `allow_general_knowledge=false`。

显式执行和查看：

```bash
.venv/bin/python -m knowledge_forge.global_search search <PROJECT_ID> "<QUERY>" \
  --graphrag-root <GRAPHRAG_PROJECT_DIR>
.venv/bin/python -m knowledge_forge.global_search show <GLOBAL_SEARCH_RUN_ID>
```

新增表为 `graphrag_community_report_runs`、`graphrag_community_reports` 和
`graphrag_global_search_runs`，由迁移 `0011_graphrag_reports_global_search` 创建。
社区报告和 Global Search 仍是 shadow：未接入默认生产查询，也未改变 P1/P2 shadow
状态。报告和查询只完成 fake-model 与 disposable PostgreSQL 工程验证；真实模型语义
质量、独立人工报告 Gold、query Gold、Local/DRIFT/Basic Search 和报告 embedding 尚未
完成，因此不能标记为生产可用。
