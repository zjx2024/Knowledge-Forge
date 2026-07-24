# Knowledge Forge P0.2：图谱策略版本与运行状态闭环

## 任务目标

请在 `zjx2024/Knowledge-Forge` 最新 `main` 上实施本补丁。基线至少包含：

- `b9b808fb3ece5ecb6ae0892e30ae150278b6d938`
- `b8cd9ac075799623d8f695f76ae05701a9c2b165`

保留现有严格 relation verifier、PostgreSQL 产品查询白名单、完整 Chunk evidence 坐标、verified surface forms 和关系审计链事务。本轮不要开始 P1 二阶段关系抽取；只关闭升级后 HugeGraph 旧边、图重建失败态、历史策略版本和关系向量一致性缺口。

先阅读 `AGENTS.md`、README 和相关实现，检查 `git status --short`，保留用户已有修改。给出简短计划后直接修改、测试、自审、提交并推送，不要停留在建议阶段。

## 已确认的残余问题

### 1. 升级后旧 HugeGraph 仍可被视为 ready

数据库迁移会隔离部分旧关系并停用向量，但不会删除 HugeGraph 旧边，也没有把已有项目的 `graph_status` 改为 `stale`。Web 只根据 `graph_status == ready` 解锁图谱按钮，直接访问 `/graph/view` 时服务端也没有验证当前图策略版本。

README 要求人工触发重建是必要运维说明，但不能代替服务端 fail-closed 门禁。

### 2. 图重建失败可能暴露空图或半图

当前 `_sync_project_graph()` 先删除项目图，再逐文档写入；如果中途失败，异常分支只把 job 标为 failed，没有把项目从旧的 `ready` 状态切换为 `failed/stale`。

### 3. 历史无当前策略版本的 `llm_verified` 仍可发布

P0.1 迁移只降级旧 `rules_verified`。旧版 `llm_verified` 可能没有 `relation-verifier-p0-v1` policy metadata，也没有当前 surface-form、ASCII 边界和完整协议保证。

### 4. 孤儿/旧关系向量可能占用 Top-K

最终 `Service.search()` 的数据库白名单能阻止数据泄漏，但 relation vector 初始 Top-K 仍可能被孤儿或非当前策略向量占据，降低已发布关系召回。

## 必须实现

### A. 策略版本必须成为发布条件，而不只是迁移备注

集中定义当前关系验证策略版本，并将默认产品查询的发布条件改为防御式规则：

- `manual_verified` 可按明确的人工审核策略处理；如果人工审核也有版本字段，应使用相应版本。
- `rules_verified` 和 `llm_verified` 只有在 `metadata.verification_policy_version == 当前版本` 时才可发布。
- 所有缺失、未知或旧策略版本默认不可发布。

不要只依赖一次性 UPDATE；即使迁移遗漏一行，产品查询也必须 fail closed。

数据库迁移应幂等地：

- 将缺少当前策略版本的旧 `rules_verified` 和 `llm_verified` assertion/candidate 降级为 `legacy_unverified`；
- 在 metadata 中保留 `legacy_verification_status` 和迁移原因；
- 停用所有不满足当前发布谓词的 relation vectors；
- 清理不存在有效 `relation_assertions` 的孤儿 relation vectors；
- 不删除审计事件或用户数据。

审核专用查询仍可读取旧状态，但必须显式调用 audit-only 方法。

### B. 增加图谱策略版本和状态机

为项目增加等价字段，具体命名遵循项目约定：

- `graph_policy_version`
- 可选 `graph_generation` / `active_graph_generation`
- `graph_status` 至少支持：`empty | stale | rebuilding | ready | failed`

状态流转：

```text
旧策略或策略缺失 -> stale
提交重建任务      -> rebuilding
完整重建成功      -> ready + current graph policy version
重建任何阶段失败  -> failed 或 stale，绝不能保持 ready
```

要求：

- 启动迁移发现项目已有图但 policy version 不匹配时，将项目标为 `stale`；新空项目保持 `empty`。
- `submit_graph_build()` 创建任务时或 worker 开始时，将项目原子切为 `rebuilding`。
- 成功处理全部文档后，才设置 `ready` 和当前 policy version。
- purge、任一文档写入、prune 或最终更新失败时，将项目设为 `failed/stale`，记录错误；重试可以安全恢复。
- 同一项目不得并发运行 ingestion、graph rebuild 或其他会改图的任务。复用现有 active-job/项目锁约束。
- 项目文档后续重新入库导致图内容过期时，应使用现有流程正确进入 building/rebuilding 状态，不能在处理中继续宣称 ready。

最低实现可以继续原地 purge+rebuild，但必须通过状态门禁阻止用户看到半图。若在现有 HugeGraph 物理模型下能低风险实现 generation/blue-green 构建后切换，可采用；不要为此引入未经验证的大型基础设施。

### C. 所有图访问必须服务端校验

不能只禁用前端按钮。

- `/graph/view`、生成 Hubble URL 的服务方法以及其他图查询入口，必须验证：
  - 项目存在；
  - `graph_status == ready`；
  - `graph_policy_version == 当前图策略版本`。
- 不满足时返回明确的 409/业务错误，例如“图谱需要按当前验证策略重建”，不能返回旧 Hubble URL。
- 图重建进行中或失败时，首页展示真实状态，并提供重试入口；不要显示 ready。
- PostgreSQL 文本/实体/Chunk 检索可以按现有能力继续工作，但图关系消费必须遵守上述门禁。

### D. HugeGraph 重建必须可验证收敛

- 显式项目图重建仍须先移除旧项目图贡献，再只写当前发布谓词允许的关系。
- 重建成功后比较 PostgreSQL 已发布关系数与本次写入/图中项目关系数；不一致时任务失败，不得设置 ready。
- 失败后再次执行必须可安全收敛，不产生重复边。
- 如果采用原地重建，catch/finally 必须保证项目不是 ready。
- 不要在应用 bootstrap 中静默执行耗时的全图重建；启动只做数据库迁移和 stale 标记，重建由持久任务显式执行。

### E. 关系向量查询直接遵守发布策略

- relation vector 检索 SQL 本身应关联有效 `relation_assertions`，验证项目、未删除状态、发布状态和当前 policy version。
- 不要先让旧/孤儿向量占据 Top-K，再完全依靠 `Service.search()` 末端丢弃。
- 保留 `Service.search()` 的最终数据库二次过滤，形成纵深防御。
- assertion 从可发布降级后，旧向量必须在同一数据库事务中停用；恢复为可发布后可重建向量。

### F. 澄清关系审计事务与向量失败语义

当前 `candidate + verification event + assertion` 使用一个 PostgreSQL 事务，这一点保留并准确命名为“关系审计链事务”，不要宣称整个 Chunk 的实体、向量和外部图写入都具备跨系统原子性。

处理 relation embedding 失败时必须选择并落实一致语义：

- 推荐：关系断言是权威数据，向量只是可重建索引。向量写失败应记录独立的 retryable indexing failure/outbox，不把语义已验证的关系改成未验证，也不要让 service 同时报告“Chunk 抽取失败”却随后把关系写入 HugeGraph。
- 或者：在索引成功前将 assertion 标为明确的 pending publication 状态，并保证 HugeGraph/搜索不发布；索引完成后原子转为发布状态。

采用与现有架构最小、最清晰的一种，并增加失败注入测试。不得吞掉错误或制造“日志失败但实际已发布”的矛盾。

### G. 小范围证据 hardening

不要全面实施 P2 span-first，但修复两个低成本一致性问题：

- 多词 ASCII surface form 也必须使用首尾字符边界，避免 `US Bank` 命中 `BUS Banking`。
- `llm_verified` 的 description 不得继续信任未经过 verifier 的抽取器自由文本。对所有成功关系使用保守、与最终 source/type/target 一致的规范描述，或扩展严格 verifier schema 返回并验证 corrected description；不能只在 reversed/type changed 时修正。

未经身份验证的 raw aliases 仍不得参与发布。若 alias 仅仅“在全文另一处出现”但不能通过现有 `credible_aliases`/缩写规则证明与 canonical name 兼容，则不要把它加入可发布 surface forms。

## 必须增加的测试

1. 历史 `llm_verified` 和 `rules_verified` 缺 policy version 时均被降级；`manual_verified` 按明确策略处理。
2. 产品查询即使在未运行迁移的构造数据上，也拒绝 policy version 不匹配的 publishable status。
3. 升级已有 `graph_status=ready` 项目后，状态变为 stale，直接 `/graph/view` 被服务端拒绝。
4. graph rebuild 提交/开始时状态为 rebuilding；成功后才 ready 且 policy version 正确。
5. 分别在 purge 后、第一篇文档后、最后一致性检查处注入失败；项目均不得 ready，Hubble URL 不可获取，重试后能成功收敛。
6. 图中关系数量与 PostgreSQL 当前已发布关系数不一致时，任务失败。
7. 孤儿、旧策略和非发布 relation vectors 不进入 relation Top-K；合法向量不会被它们挤占。
8. assertion 从已发布降级后，向量在同一事务中停用。
9. relation embedding 写失败时，任务日志、assertion 发布状态和 HugeGraph 最终行为符合所选语义。
10. 多词 ASCII 边界测试：`US Bank` 不得命中 `BUS Banking`。
11. 普通 `as_is`、未改 type 的 supported 关系也不能发布抽取器幻觉 description。
12. 使用真实 disposable PostgreSQL 调用实际 `Database` 方法：
    - `SCHEMA_SQL` 连续执行两次；
    - 迁移 rules/llm 状态和孤儿向量；
    - 在 candidate/event 成功后让 assertion 触发约束错误，断言关系审计三张表全部回滚；
    - 成功链的三张表具有相同非空 `ingestion_run_id`。

真实 PostgreSQL 测试继续使用独立临时 schema/database，禁止连接或清理真实用户库。如果缺少 `KNOWLEDGE_FORGE_TEST_POSTGRES_DSN`，不要伪造通过；最终报告必须将 P0.2 标记为“尚未完成真实数据库验收”。

## 验收标准

- 旧策略关系在未迁移、迁移中和迁移后均无法从产品搜索或图入口泄漏。
- 图策略不匹配、重建中或重建失败的项目均无法获取 Hubble URL。
- 图重建失败不会留下仍标 ready 的空图/半图。
- 历史无当前 policy 的 llm/rules 关系和向量均被隔离。
- relation vector Top-K 本身遵守当前发布策略。
- 关系审计链事务在真实 PostgreSQL 中验证回滚。
- 完整单元测试、真实 PostgreSQL 集成测试、编译检查和 `git diff --check` 通过。

## 最终必须使用本地 GitHub App 提交并推送

只有前述检查完成后，执行以下交付流程：

1. 检查 `git status --short` 和完整 diff，确认不覆盖用户修改。
2. 只暂存本任务相关实现、测试和 README/迁移说明。不要提交 `Knowledge-Forge-Codex-P0*.md` 等执行指令文件，除非用户明确要求纳入仓库。
3. 使用清晰的提交信息，例如：

   ```text
   fix: close graph verification policy lifecycle
   ```

4. 使用当前环境已经登录的本地 GitHub App/凭据，将提交推送到当前远程分支。禁止 force-push、禁止改写历史、禁止删除远程分支。
5. 推送后读取远程分支头，确认远程 commit SHA 与本地 HEAD 一致。
6. 最终报告必须列出：本地 commit SHA、远程分支、push 结果、测试命令与数量、真实 PostgreSQL 集成测试是否运行、未解决风险。

如果测试失败或真实 PostgreSQL 门禁未完成，可以提交并推送明确标注为未完成的修复分支，但不得声称 P0.2 已验收通过，也不得把失败测试删除或放宽。
