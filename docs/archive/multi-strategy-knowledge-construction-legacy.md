# 多策略知识构建实施计划

> 本文是可随实现持续更新的设计基线。代码、迁移与测试发生变化时，应同步更新“完成状态”和兼容说明。

## 1. 目标

Knowledge Forge 需要在不重启服务、不重复保存源文件的前提下，对同一项目、同一文档运行不同的实体关系抽取与归一策略，并允许用户比较各策略的状态、日志和知识图谱。

首期提供两种策略：

| 策略键 | 页面名称 | 定位 |
|---|---|---|
| `stable` | 稳定基线 | 当前正式抽取、关系复核与实体归一流程 |
| `enhanced_shadow` | 增强实验 | P1 关系处理与 P2 实体归一的隔离实验流程；只用于比较，不改变生产策略或激活门禁 |

P1/P2 继续保持 shadow。页面允许构建和查看实验结果，不表示策略已经 `production_approved`，也不会自动成为项目默认生产策略。

## 2. 数据分层

### 2.1 共享层

以下数据与构建策略无关，只处理或存储一次：

- `documents` 源文档及上传状态；
- 文档结构解析结果；
- `sections`、`chunks` 和向量索引；
- 当前项目的逻辑 Schema；
- 解析器、分块策略与源文件过程文件。

当文档尚未生成共享底座时，任一策略的“开始构建”都会先完成结构解析、分块、向量索引和逻辑 Schema 准备。

### 2.2 策略层

从实体关系抽取、归一、关系复核和图谱同步开始按策略隔离：

- 每个文档、每个策略拥有独立构建状态、错误、统计、最新任务和完成时间；
- 每个项目、每个策略拥有独立图谱代次、状态、错误和 HugeGraph namespace；
- 任务必须携带 `strategy_key`，日志按策略查询；
- HugeGraph 顶点和边必须携带 `kf_strategy_key`，自定义 ID 包含策略 namespace；
- 切换策略不会修改另一策略的状态或图谱。
- 稳定基线允许不同文档按执行器顺序排队；项目级增强实验与任何稳定基线任务互斥。
- 任一稳定文档重建都会使该项目完整的增强快照失效，避免显示混合修订结果。

## 3. PostgreSQL 模型

新增 `document_strategy_builds`：

- 唯一键 `(document_id, strategy_key)`；
- `status`: `uploaded/queued/processing/indexed/failed/cancelled/deleted`；
- `latest_job_id`、`error_message`、`stats`；
- `content_revision_id` 指向该策略使用的不可变内容修订；
- `started_at/completed_at/updated_at`。

新增 `project_strategy_graphs`：

- 唯一键 `(project_id, strategy_key)`；
- 独立 `namespace_id/generation/ready_generation/status/error`；
- 保存策略政策摘要与更新时间。

为 `jobs` 增加 `strategy_key`。现有历史任务回填为 `stable`。

共享内容修订仍由 `documents.active_revision_id` 管理。实验策略引用同一份完整修订，但实验实体、关系及图谱产物写入其隔离命名空间，不抢占正式 active revision。

## 4. 操作流程

1. **上传**：只保存文件并创建 `documents`，页面状态为“已上传”。
2. **选择策略**：页面顶部构建方案下拉框决定列表中展示的策略状态、任务日志和图谱入口。
3. **构建**：点击文档行的“构建知识”按钮。
   - 共享底座不存在：先执行结构解析、分块、向量索引；
   - `stable`：执行正式抽取、归一、复核并同步稳定图谱；
   - `enhanced_shadow`：自动准备或复用完整共享内容运行隔离实验，不激活生产策略，并同步实验图谱。
4. **查看**：日志、统计、按钮可用性和 Hubble 查询都按当前策略读取。
5. **删除**：源文档删除时，取消所有策略任务，软删除全部策略状态，并从各策略 namespace 清理其贡献。

## 5. 失败与恢复

- 单个知识块的模型截断先降级为 entity-only，再执行小型 relation-only 补抽取；不能因一次超长输出丢失整个文档。
- 所有 `ON CONFLICT` 必须与迁移创建的唯一索引完全一致。
- 内容发布字段由数据库方法统一拥有，调用者不得重复更新 `content_version`、`active_revision_id` 或 `updated_at`。
- 失败策略只标记自己的 `document_strategy_builds` 和 `project_strategy_graphs`，不污染另一策略。
- 服务重启后根据 PostgreSQL 作业和策略状态恢复显示，不依赖进程内存。

## 6. 页面变更

- “上传解析”拆为“上传文件”；上传成功后仅进入文件列表。
- “查看知识构建日志”左侧增加“构建方案”下拉框。
- 文档行提供“构建知识”；状态按所选方案显示。
- “查看知识图谱”携带项目和策略 namespace，并直接执行对应查询。
- 实验方案显示“仅供对比，未生产激活”的固定提示。

## 7. 验证门槛

- 迁移可在已有 0011 数据库上原位应用并幂等启动；
- 两个策略状态和图谱 namespace 不互相覆盖；
- 上传不自动入库，构建才创建任务；
- 正式流程不再出现 content version 重复赋值和错误 conflict target；
- 模型截断可降级恢复，关系补抽取使用较小输出协议；
- targeted tests、完整无外部服务测试、`py_compile`、`pip check`、`git diff --check` 全部通过；
- 本地服务重建后 `/health` 可用，手工验证同一文档两种方案的状态、日志和图谱入口。

## 8. 完成状态

- [x] 现有失败根因取证
- [x] 内容发布字段去重防御
- [x] 模型截断降级与 relation-only 补抽取
- [x] 0012 多策略数据库迁移
- [x] 策略级服务编排与图谱隔离
- [x] 上传/构建拆分和页面交互
- [x] 自动化回归与本地运行验证
