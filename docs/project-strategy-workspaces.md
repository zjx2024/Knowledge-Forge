# 项目与知识图谱策略固定关联

## 目标

一个项目只属于一种知识图谱构建策略：

- `knowledge_forge`：用户界面名称为 **Knowledge Forge**；
- `graphrag_native`：用户界面名称为 **GraphRAG Native**。

历史输入键 `enhanced_shadow` 只在旧请求兼容层归一为 `knowledge_forge`，新项目和
新构建不再写入该键。策略不是页面筛选条件，也不能通过 URL 参数、构建按钮或
普通 API 改变。

## 生命周期

1. 新项目在创建事务中同时写入项目记录和 `bound` 策略记录。
2. 0024 迁移前且尚未绑定的项目由首页要求用户一次性确认，不根据历史查询参数猜测。
3. 绑定通过 generation CAS 完成；绑定后应用接口和 PostgreSQL trigger 都拒绝修改。
4. 已有其他策略产物不自动删除，只是不再出现在固定策略工作区中。

## 页面与路由

- `/`：公共项目首页，只负责项目创建、项目选择和旧项目首次确认。
- `/projects/{id}/knowledge-forge`：Knowledge Forge 专属工作区。
- `/projects/{id}/graphrag-native`：GraphRAG Native 专属工作区。

旧的 `/?project=...&strategy=...` 只根据数据库绑定结果跳转，查询参数不能决定策略。
构建、准备、统计、查询和查看接口均在服务端检查项目绑定。

## 两套准备与构建契约

### Knowledge Forge

源文件 → 结构块 → Chunk → pgvector → 逻辑 Schema → 实体/关系抽取与归一
→ 隔离图物化。

项目设置包括解析器、分块策略、模型、上下文预算和单 Chunk 上限。

### GraphRAG Native

源文件 → 解析器结构块 → 冻结 Native Documents → GraphRAG 3.1.1 Standard
workflow → TextUnits、原生实体/关系、社区、社区报告、GraphML 和 LanceDB bundle。

准备源文档时不会生成 Knowledge Forge Chunk、pgvector 或逻辑 Schema。TextUnit
参数由版本化的 Native 运行配置拥有；当前页面只读展示
`900 tokens / 100 overlap / o200k_base`。

## GraphRAG Native 过程产物

页面只提供经过白名单和分页限制的只读查看：

- 准备阶段生成的 Native Documents；
- `input/documents.parquet`；
- `kf-manifest/source-map.parquet`；
- TextUnits、实体、关系、社区和社区报告 Parquet。

LanceDB 不开放文件浏览；查询仍通过受控 Native query worker 完成。所有查看接口
只读取当前 fresh、独立验证通过的 immutable bundle。

## 兼容与安全

- 0024 只增加绑定列、约束、索引和不可变 trigger，不删除项目或历史产物。
- Native 源 revision 激活不推进旧图 generation，也不退休历史规范实体。
- Native 不调用 Knowledge Forge Chunk、pgvector、实体关系管线或 HugeGraph。
- Knowledge Forge 接口拒绝 Native 项目，Native 接口也拒绝 Knowledge Forge 项目。
