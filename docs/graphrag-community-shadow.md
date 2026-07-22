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

只有正式关系实际引用的规范实体会进入 BYOG 节点。实体 `title` 使用项目内唯一的
`entity_key`，显示名称、类型和描述放入 `description`。Chunk 来源使用稳定的
`chunk:<revision_id>:<chunk_key>` 标识。

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

当前未接入 `create_community_reports`、报告 embedding、Global Search、Local Search、
DRIFT Search 或 Basic Search，也未改变 P1/P2 shadow 状态。
