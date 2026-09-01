# Knowledge Forge 命名与兼容边界

> 更新时间：2026-09-01

## 当前术语

| 当前名称 | 含义 | 主要代码 |
| --- | --- | --- |
| grounded entity | 带精确 occurrence、来源和保守归一结果的实体快照 | `entity_runtime.py`、`grounded_entities.py` |
| grounded relation | 经候选、验证、归一和发布审计的关系事实 | `relation_extraction.py`、`relation_runtime.py`、`grounded_relations.py` |
| evidence context | 将渲染文本映射回解析块、页码和结构角色的证据上下文 | `evidence_context.py`、`grounded_context.py` |
| relation pipeline | 旧兼容构建路径中的有界候选分类流水线 | `relation_pipeline.py` |
| Knowledge Forge | 当前项目级知识图谱构建策略，键为 `knowledge_forge` | `construction_strategies.py` |
| GraphRAG Native | 独立 GraphRAG 3.1.1 项目级策略，键为 `graphrag_native` | `graphrag_native_*` |
| knowledge agent | 两种构建策略之上的统一只读问答层；当前显式绑定一个知识库 | `knowledge_forge/agent/` |
| evidence item | 统一查询证据，绑定项目、策略、快照、来源和运行级稳定键 | `agent/contracts.py` |
| conversation context | 只用于指代消解的最近轮次与滚动摘要，不是回答证据 | `agent/context.py` |
| layered policy | 访问、输入、对话、检索、执行和输出六层确定性决策协议 | `agent/policy.py` |

新代码、函数、脚本、日志和文档应使用这些名称，不再创建包含 `shadow` 的新应用层标识。

## 必须保留的历史标识

以下内容已经写入用户数据库、HugeGraph、运行快照或不可变评估记录，不能通过普通重构直接改名：

- PostgreSQL 表、列、约束和索引，如 `entity_shadow_*`、`shadow_relation_*`、`relation_shadow_*`；
- 已执行迁移版本，如 `0014_shadow_relation_v2_ledger`；
- 历史策略输入别名 `enhanced_shadow`；
- 运行 ID、规范实体 ID、上下文 ID 和 HugeGraph ID 的历史前缀；
- Prompt、Parser、Schema 和策略版本值中的历史版本字符串；
- 已持久化 JSON 字段，如 `shadow_context_*` 和 `shadow_relation_schema_overlay_version`。

这些标识只属于存储/协议兼容层。修改它们会导致旧快照指纹变化、模型调用缓存失效、外键/索引断裂或已有图节点不可寻址。

## 代码约束

1. 业务代码通过当前命名的 Python API 访问历史物理表；SQL 中保留原表名。
2. 新迁移不得重写或删除历史表；如未来需要物理改名，必须采用新增表、双读、回填、校验、切换和独立回滚的正式迁移方案。
3. 不为美化命名改变不可变 Prompt/策略版本；语义改变必须创建新版本并按新鲜度规则重建。
4. `enhanced_shadow` 只作为旧请求输入别名归一到 `knowledge_forge`，新记录不得继续写该策略键。
5. 测试中出现旧名称时，应明确是在验证物理兼容合同，而不是继续传播旧产品术语。
6. 智能体新代码不得把 KF/GR 内部 ID 或得分当成跨策略公共标识；必须先转换为统一证据协议。
7. LangGraph checkpoint 是有界流程状态，`agent_messages` 才是完整会话事实来源；两者不能互相改名或替代。

## 已完成迁移

- 当前实体、关系、证据上下文、Schema overlay 和 replay 模块已采用现行文件名；
- 运行时类、公共函数、配置项和策略常量已采用现行名称；
- 评测导出、验证和恢复脚本已采用现行名称；
- 测试文件与夹具按当前能力命名；
- 旧一次性 pilot 和未使用的原型入口已删除；
- `artifacts/` 改为可重复生成且不进入 Git。
- 统一问答模块直接采用 `agent`、`evidence`、`citation`、`context` 和 `policy` 命名，没有引入新的 `shadow` 应用层标识。
