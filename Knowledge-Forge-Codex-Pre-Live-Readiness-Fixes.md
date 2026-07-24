# Knowledge Forge：进入真实测试前的完整修复、验证与普通 Git 推送任务

## 0. Codex 必须执行的总指令

请在 zjx2024/Knowledge-Forge 的最新 origin/main 上连续完成本文，不要只给计划、建议、伪代码或差异说明。先复现并锁定问题，再实现、测试、自审、分阶段提交和普通 Git 推送，直到达到本文定义的 engineering_complete 和 isolated_live_test_ready，或遇到本文允许的真实外部阻塞。

本文是对以下任务文档的审查修订和最终补充：

- Knowledge-Forge-Codex-P0-Task.md
- Knowledge-Forge-Codex-P0.1-Review-Fixes.md
- Knowledge-Forge-Codex-P0.2-Graph-State-Task.md
- Knowledge-Forge-Codex-P1-P3-Complete-Program.md

P0～P0.2 的 fail-closed、发布白名单、审计历史和图状态约束继续有效。若旧任务文档与本文冲突，以更严格、更安全、可证明的本文为准。

执行要求：

1. 读取 README、数据库模型、服务入口、抽取/归一/关系管线、HugeGraph、迁移、评估、benchmark、Docker 和全部测试。
2. 用代码搜索确认本文列出的符号是否仍存在；远程代码如已变化，按语义定位，不机械依赖行号。
3. 每阶段先增加会失败的 characterization/regression test，再修实现。
4. 每阶段运行 targeted tests、核心回归、自审 diff，并创建独立本地 commit。
5. 全程使用本地普通 git；不使用 GitHub App，不 force-push。
6. 任何 skipped 都单独列出原因，不能把 skipped 计为 passed。
7. 不连接、不修改、不 purge 用户现有 PostgreSQL、HugeGraph、Ollama 数据。
8. 缺少外部服务时继续完成 fake/disposable 工程验证，不得无限等待。
9. 缺少独立人工 Gold 时 treatment 保持 shadow，不得冒充 production_approved。
10. 任务结束必须推送工作分支；满足本文全部门禁时再普通 fast-forward 推送 main。

## 1. 本次审查基线与结论

审查时远程 main 的提交为：

    a4afc9f989e2df2133fdf64eabee50543fb30b70

开始执行时必须先 fetch，并以实际 origin/main 为准，不得为了回到上述 SHA 而覆盖后续提交。

当前结论不是“可以直接使用业务数据测试”。代码已具备 P0～P3 的许多组件，但至少有以下 release blocker：

1. 文档重入库仍可能硬删被审计表外键引用的行，默认路径下第二次入库可能失败或破坏历史。
2. Ollama 的 max_tokens 被混作输入上下文预算，且未正确设置输出上限 num_predict。
3. 关系 dense Chunk 处理缺少可恢复的 deferred/retry 闭环和严格 evidence span 证明。
4. P2 span-first 实现没有真正接入端到端 treatment；现有实体 shadow 主要重放 control 结果，不是独立比较。
5. policy snapshot 有表和状态，但缺少能从原始 Chunk 构建完整不可变快照的可恢复 orchestrator。
6. P3 指标和 gate 可能忽略缺失 mention/预测，且 gate 接受调用方汇总指标，不能证明来自冻结 artifact。
7. HugeGraph purge 只按整数 project_id 隔离，测试数据库 ID 与业务图 ID 碰撞时可能误删。
8. graph ready 主要依赖数量，无法排除等量但错端点、错方向、错 label 或混入其他代次。
9. 启动时巨型 DDL、破坏性 DROP 和重回填不具备版本化迁移安全性。
10. 默认网络暴露、弱默认凭据、构建上下文和 CI 仍不足以支撑共享实际测试。

只有这些问题关闭后，才可以进入“隔离环境真实服务测试”；production_approved 还需要独立人工 Gold、冻结 holdout、真实 shadow/canary 和明确运维批准。

## 2. 三个不可混淆的完成状态

    engineering_complete:
      本文全部代码、迁移、fake/disposable 测试、失败注入、恢复、CI 和文档完成。

    isolated_live_test_ready:
      engineering_complete；
      preflight 能证明 PostgreSQL、Ollama、Embedding、reranker、HugeGraph 均为隔离测试资源；
      destructive graph 测试有 namespace、sentinel 和双开关；
      可以安全开始使用合成或已脱敏语料做真实 E2E。

    production_approved:
      isolated_live_test_ready；
      独立人工复核 Gold、冻结 holdout、真实模型评估、隔离图 E2E、shadow/canary、
      安全评审、备份恢复和所有 release gate 均通过；
      treatment policy 才允许激活到业务项目。

Codex 本轮必须尽可能达到前两个状态。没有真实证据时只能报告 production_approved=false。

## 3. Git 和工作区保护

### 3.1 开始取证

在仓库根目录执行：

    set -euo pipefail
    git rev-parse --show-toplevel
    git status --short --branch
    git remote -v
    git rev-parse HEAD
    git rev-parse HEAD^{tree}
    git diff --name-status
    git ls-files --others --exclude-standard
    git fetch --prune origin
    git rev-parse origin/main
    git rev-parse origin/main^{tree}
    git rev-list --left-right --count HEAD...origin/main
    git log --oneline --decorate -n 20 origin/main

记录开始时本地/远程 commit、tree、ahead/behind、tracked 修改和未跟踪文件。

禁止：

    git reset --hard
    git clean
    git stash --include-untracked
    git add .
    git add -A
    git push --force
    git push --force-with-lease

### 3.2 创建隔离工作分支

若 tracked 工作区干净，或只有本文等未跟踪任务文档：

    WORK_BRANCH=codex/pre-live-readiness
    git switch -c "$WORK_BRANCH" origin/main

若分支已存在，使用唯一后缀并保存实际名称。若存在不属于本任务的 tracked 用户修改，不覆盖、不提交、不 stash；创建独立 worktree：

    WORK_BRANCH=codex/pre-live-readiness-2
    git worktree add ../knowledge-forge-pre-live-readiness \
      -b "$WORK_BRANCH" origin/main

进入新 worktree 后重新确认 status 和 HEAD。任务文档默认不提交。每次只显式暂存本阶段文件：

    git add -- path/to/file1 path/to/file2
    git diff --cached --check
    git diff --cached --name-status
    git diff --cached

## 4. P0～P0.2 永久不变量

任何修复都不得破坏：

- 关系只有当前允许的 policy/version 和 publishable status 才能进入产品查询、向量与 HugeGraph。
- verifier 非法 JSON、漏项、未知 ID、证据不合法、方向错误、超时或异常一律 fail closed。
- verification_failed、verification_missing、uncertain、rejected、legacy/旧 policy 不发布。
- relation candidate、verification event、mention-level assertion 保留不可变审计。
- entity merge/split 只改变版本化 assignment/projection，不能改写旧 assertion 让其看似验证了新端点。
- 当前策略必须在 vector Top-K 的 LIMIT 之前过滤。
- graph 非 ready 或 generation/policy 不匹配时，view 返回 409。
- indexing/outbox 失败不降低已经验证的关系状态。
- 任何数量、内容、端点、policy、generation 不一致都不能把图标记 ready。
- 不用性能优化换取静默漏候选、宽松证据、全表 Python 两两比较或长期 deferred。

## 5. 阶段 A：基线、依赖图和失败测试

### 5.1 环境和测试基线

执行并保存完整输出摘要：

    python3 --version
    test -x .venv/bin/python && .venv/bin/python --version || true
    test -x .venv/bin/pip && .venv/bin/pip check || true
    docker --version || true
    docker compose version || true
    rg -n "TODO|FIXME|pass$|NotImplemented|skip|xfail" \
      knowledge_forge tests scripts benchmarks .github README.md || true
    rg -n "replace_document_content|relation_shadow_runs|policy_snapshot|SpanFirst|json_chat|max_tokens|num_predict|purge|graph.*ready|schema_migrations" \
      knowledge_forge tests scripts benchmarks README.md || true

若 .venv 不存在，按仓库现有说明创建，不擅自升级主版本。安装必须使用仓库锁定/约束文件；如果目前没有可复现锁定，阶段 I 补齐。

默认无外部服务测试：

    .venv/bin/python -m pytest -q -ra --strict-markers \
      -m "not integration and not live_model and not hugegraph_integration"
    .venv/bin/python -m py_compile knowledge_forge/*.py server.py
    .venv/bin/pip check
    git diff --check

记录 passed/skipped/failed 和耗时。基线失败不能删除测试或降低断言；先判断是已有失败还是本阶段回归。

### 5.2 建立数据库删除依赖图

使用 PostgreSQL catalog 或从 DDL/SQL 生成机器可检验的外键依赖图，至少覆盖：

- documents
- chunks
- entity_mentions
- relation_assertions
- mention_relation_assertions
- canonical_relation_projections
- entity_assignments
- entity_type_assertions
- relation_shadow_runs
- policy snapshot 相关表
- outbox/event 表

为 replace_document_content、delete_document、retry/reingest、restore 分别列出：

- 当前会 INSERT/UPDATE/DELETE 的表；
- 被哪些外键引用；
- ON DELETE 行为；
- 哪些是不可变审计，哪些是当前投影；
- 事务边界；
- 失败后的可重试状态。

新增一个真实 disposable PostgreSQL characterization test，证明当前默认抽取路径在同一 document 上连续重入库至少三次、第二次失败注入后恢复、服务重启后继续。修复前该测试应暴露硬删除/外键或历史丢失问题。

### 5.3 本阶段建议新增测试

名称可按项目风格调整，但覆盖不能减少：

    tests/test_document_content_revisions.py
    tests/test_ollama_token_contract.py
    tests/test_relation_shadow_recovery.py
    tests/test_entity_shadow_independence.py
    tests/test_policy_snapshot_orchestrator.py
    tests/test_release_gate_artifacts.py
    tests/test_hugegraph_safety_guard.py
    tests/test_graph_manifest_validation.py
    tests/test_migrations.py
    tests/test_pre_live_e2e.py

提交：

    test(readiness): reproduce pre-live release blockers

## 6. 阶段 B：用内容修订替代破坏性重入库

### 6.1 数据模型

不要给审计表加 CASCADE 来掩盖问题，也不要硬删旧 assertion/event/assignment。实现文档内容修订，具体命名可适配仓库，但语义必须具备：

1. document_content_revisions：
   - 稳定 revision_id；
   - project_id、document_id；
   - ingestion_run_id；
   - content hash、source/config/policy hash；
   - building、active、retired、failed 等状态；
   - created_at、activated_at、retired_at、failure；
   - 同一 document 最多一个 active revision 的唯一约束。
2. documents 保存或可确定查询 active_revision_id。
3. chunks、mentions、relation candidates/events/assertions、entity assignments/type assertions、shadow items 和 canonical projection 能关联 revision_id 或等价不可变 lineage。
4. 历史 revision 只退休，不硬删除；合规清理另做显式 retention job，并默认关闭。
5. active 产品查询只读取 active revision + 当前 policy；审计 API 可显式查看历史。

数据库约束必须真正支持历史并存，不能靠 upsert 覆盖旧代次：

- revision 表要求 UNIQUE(document_id, content_version) 和 UNIQUE(document_id, ingestion_run_id)；
- revision 表提供 (revision_id, project_id, document_id) 复合唯一键；
- 事实表用复合 FK 证明 revision 属于同一 project/document；
- Chunk 事实用 (chunk_id, revision_id) 或等价复合 FK 防串代；
- sections、chunks、relation assertions、entity type assertions 等旧唯一键加入 revision_id；
- vector/outbox 若同时包含文档对象和项目级对象，分别使用 revision_id IS NOT NULL 与 IS NULL 的 partial unique index，避免 NULL 语义产生重复；
- 同一 ingestion_run_id + 同一 manifest 才可幂等返回；同一 run + 不同 manifest 必须拒绝。

迁移/测试必须故意写入“项目 A 的事实 + 文档 B 的 revision”和“Chunk C + revision D”，并确认由 PostgreSQL 约束拒绝，而不是仅靠 Python 检查。

### 6.1.1 最小 migration 基础必须提前

阶段 B 的 revision schema 不能先塞回巨型 bootstrap、等阶段 I 才迁移。开始实现 revision 前，先落地最小 schema_migrations、checksum 和 PostgreSQL advisory migration lock；所有 revision DDL/backfill 从第一次实现起就通过版本化 migration。阶段 I 再补 plan/status、分批大 backfill、并发索引和完整运维体验。

旧数据 backfill 若发现早期 candidate/event 的 chunk_id 已失联：

- 不得伪归到当前 active revision；
- 创建明确的 legacy orphan/superseded revision；
- 保留原始 chunk ID/payload；
- 标记 provenance_ambiguous；
- 产品查询永远不使用该 revision；
- FK 校验和审计 digest 仍可通过。

### 6.2 写入协议

实现短事务、两阶段构建：

1. 短事务创建 building revision 和 ingestion_run。
2. 在事务外做 Chunk、模型、embedding/reranker 调用。
3. 每个 Chunk 使用短事务写不可变事实，并带 revision_id/run_id。
4. 所有 Chunk 和必须 outbox 达到终态后，在一个短事务中：
   - 加 document 级 advisory lock 或行锁；
   - 再次校验 run 仍为预期；
   - 原子激活新 revision；
   - 退休旧 active revision；
   - 更新 documents.active_revision_id；
   - 递增需要的 projection/graph generation 并置 stale。
5. 旧 worker 晚到不得覆盖较新的 active revision；使用 compare-and-set generation/run guard。
6. 构建失败保留 failed revision 和审计，不影响旧 active 内容继续服务。
7. 重试必须复用或明确 supersede run，不能生成重复发布事实。

防旧 worker 不能只放在最终激活：每次 mention、candidate、assertion、assignment/type、vector、outbox 写入都必须在 SQL 层校验 revision/run 仍处于允许写状态且 lease token 有效。late worker 只能得到明确 StaleContentRevision/LeaseLost，不能继续污染 retired/failed revision。

不得持有数据库事务等待 Ollama、reranker、Embedding 或 HugeGraph。

### 6.3 删除、恢复和投影

- 普通文档删除是逻辑删除，当前 revision 退出产品视图，审计历史仍保留。
- 恢复必须显式选择可恢复 revision 或触发新 revision，不可把旧投影静默混回当前 policy。
- canonical source_count、relation projection、vector outbox 和 graph rebuild 只基于 active revisions。
- policy snapshot 可以冻结具体 revision 集合；之后 active revision 改变不能修改旧 snapshot。
- 创建/building revision 时不得改变旧 ready generation；只有新 revision 原子激活时才递增 generation 并置 stale。失败 building revision 必须保留旧 active revision 和旧 ready 图继续服务。
- 删除必须幂等；重复删除不能重复推进 graph generation。
- 恢复或删除后同 (project_id, doc_hash) 再上传必须有确定行为；恢复后旧 verified assertion/projection 不得自动重新发布，必须经过当前策略处理和新图重建。

### 6.4 必须通过的 PostgreSQL 场景

在临时数据库/schema 中验证：

- 新文档首次入库；
- 同一文档相同内容幂等重试；
- 内容不同连续重入库三次；
- 第 N 个 Chunk 失败，旧 active 仍可读；
- 重启后恢复 building run；
- 两个并发 run 乱序完成，旧 worker 无法覆盖新 active；
- relation/assertion/assignment/type/shadow 外键全部有效；
- 旧 revision 审计仍可查询；
- active 产品搜索、统计、关系 ID 回取和 graph snapshot 不混入旧 revision；
- 删除、恢复、再次入库；
- 整个激活事务中途失败时完整回滚。
- 相同 run/manifest 幂等、相同 run/不同 manifest 拒绝；
- 每类事实的 late worker 写入均被 SQL guard 拒绝；
- 重入库、删除、恢复、迁移前后不可变审计表的主键集合及规范化 JSON payload digest 不变，而不只是“仍可查询”。

禁止只用 SQLite/fake 证明 PostgreSQL 外键与锁语义。

提交：

    fix(storage): make document reingestion revisioned and auditable

## 7. 阶段 C：修正 Ollama token/context 协议

### 7.1 问题

审查发现 relation pipeline 把输出预留值（默认约 512）传给 json_chat(max_tokens=...)；而 OllamaClient 又把该值当成整个输入上下文预算，且没有向 Ollama 发送 options.num_predict。这会造成错误截断、预算判断失真或输出上限未生效。

### 7.2 显式预算类型

删除含义模糊的 max_tokens 跨层传递，至少拆成：

- context_budget_tokens：应用允许送入模型的总输入预算；
- max_output_tokens：模型最大输出 token，对应 Ollama options.num_predict；
- safety_margin_tokens：模板、JSON、tokenizer 偏差预留；
- estimated_input_tokens：发请求前由匹配 tokenizer 或经 live 校准的保守估算器计算；
- prompt_eval_count：Ollama 响应返回的实际输入 token，用于事后校准；
- output_reserve_tokens：若预算算法需要，必须和 max_output_tokens 关系明确。

要求：

1. Prompt admission 应满足 estimated_input + max_output + safety_margin <= 已验证 effective context。
2. transport 向 Ollama 发送 options.num_predict=max_output_tokens。
3. 默认不要发送 options.num_ctx；项目当前设计是应用级预算，不应按请求制造不同 Ollama runner。显式启用时 num_ctx 必须是模型/部署级固定值，不能逐请求漂移。
4. /api/show 只用于读取模型最大 context、digest 和 Modelfile 参数；预加载模型后必须用 /api/ps 的 context_length 核验当前 runner 的有效 context。无法取得 effective context 时，使用显式保守上限或 fail closed，不能把模型最大值冒充实际 runner 上限。
5. 配置值大于 effective context 时 fail closed 或降到经过记录的安全值；显式 num_ctx 后也必须再用 /api/ps 确认。
6. tokenizer 估算器必须可替换、可观测；不确定时保守。保存 estimated_input_tokens、响应 prompt_eval_count 和 rendered-message hash。
7. 估算与 prompt_eval_count 偏差超过阈值、服务疑似截断或 done_reason=length 时结果 fail closed 并产生可诊断错误，不能用字符数除四宣称已验证。
8. 请求日志只记录 token 数、模型/digest、预算和 hash，不记录完整正文/Prompt。
9. structured JSON 错误、超时、输出截断继续走既有严格 verifier 和 bounded retry，不可放宽发布。

### 7.3 Transport 边界测试

用假 HTTP server 或 request spy 断言：

- num_predict 精确等于 max_output_tokens；
- 默认请求没有 num_ctx；
- context_budget 不会误写到 num_predict；
- 超预算在发 HTTP 前被拆批/defer；
- estimated input 位于边界内时允许发送；
- 等于/超过边界的 off-by-one 行为固定；
- /api/show 模型上限、/api/ps runner context 和配置不一致时处理正确；
- prompt_eval_count 偏差和 done_reason=length 会 fail closed；
- timeout、invalid JSON、partial output 仍 fail closed；
- relation、entity、verifier 调用都使用同一明确 contract。

README 说明每个环境变量的单位、默认值、合法范围和与 Ollama 参数的映射。

提交：

    fix(llm): separate context and output token budgets

## 8. 阶段 D：完成可恢复且可证明的 P1 关系 treatment

### 8.1 精确 evidence

每条 relation prediction 的 evidence 必须来自完整原始 Chunk 坐标：

- chunk_id、revision_id；
- start_char、end_char，半开区间；
- evidence_text；
- evidence_hash；
- subject/object mention ID 与其原始 spans；
- predicate、direction、policy/model/config versions。

验证必须满足原文 chunk_text[start:end] 与 evidence_text 精确一致并匹配 hash。重复句子或重复证据不能用第一次 str.find 猜坐标；span 必须由分句/窗口构建时携带 offset，或在有唯一可证明 occurrence 的情况下解析。找不到、多个无法消歧、越界、规范化后才匹配都隔离，不发布。

修复 sentence splitter，覆盖：

- 小数 3.14；
- 版本 v1.2.3；
- URL、域名和邮箱；
- 中英文缩写；
- 省略号；
- 引号、括号；
- 中文句号/问号/感叹号；
- 重复相同句子；
- 自然正文包含“分块:”等字样。

### 8.2 候选与 dense Chunk 闭环

不要把 overflow/deferred 当成功。实现持久化、可恢复的 relation work queue 或 outbox：

    pending -> leased -> completed
                      -> retryable_failed -> pending
                      -> terminal_failed
                      -> quarantined

要求：

- lease 有 owner、expires_at、attempt、next_attempt_at；
- 服务重启或 worker 死亡后可回收；
- 同一 snapshot/revision/chunk/work item 幂等；
- 单次只取符合 token 预算的 candidate batch；
- 超预算先按完整句子/结构窗口拆分，再拆候选批次；
- 本批次处理得下的必须处理，只 defer 剩余部分；
- overlap 和跨窗候选有去重 ID；
- bounded attempts 后进入明确终态并计入质量 false negative；
- snapshot complete 前所有 work item 必须为完成或明确失败；有 pending/deferred 不能宣称成功。

candidate_id/work_item_id 必须由稳定输入和 policy 版本确定性生成。并发 worker 使用 SKIP LOCKED 或等价有界租约；不持长事务跑模型。

### 8.3 复杂度与质量

- 普通关系规划目标 O(E log E + 局部窗口候选)。
- 若一句内 E 个实体确实两两可能有关，输出下界为 Ω(E²)；允许有界流式批次，禁止 candidates[:N] 静默丢失。
- Prompt 中相同 context 只出现一次；候选用紧凑 ID 表。
- 所有 raw_pairs、blocked_pairs、admitted、classified、verified、published、deferred、retried、failed 必须可核对。
- gold positive 落在 blocked/deferred/timeout/failed 中计 false negative。
- control 与 treatment 使用同一 frozen input universe。

### 8.4 独立 relation-only shadow

提供 relation_only snapshot：

- 使用冻结的 Gold 或 control-approved entity mention spans/IDs；
- 从原始 Chunk 独立运行 treatment candidate planner、classifier、verifier；
- 不能把 control relation candidates/predictions 重命名后当 treatment；
- 所有结果写 shadow namespace，正式 assertion/vector/search/graph 零写入；
- 能输出逐项 discordance、候选漏失、证据错误、token/latency/call counts。

测试证明：

- dense Chunk 拆批后无候选丢失；
- 中途失败/重启最终收敛；
- crash/resume 对同一 work item 复用已经持久化的 terminal result；固定 fake response 下 artifact hash 一致；
- false success 不可能；
- formal tables 的发布行、向量、图 outbox 在 shadow 前后完全不变；
- evidence 坐标对重复文本准确。

不要要求两次全新的 live LLM inference 字节级相同。持久化稳定 input/run fingerprint，并为每次 attempt 保存不可变 output hash；首个通过 compare-and-set 接受的 terminal attempt 决定 snapshot，晚到 worker不得覆盖。不同 live 输出必须留在审计中，不能伪造成相同 hash。

提交：

    fix(relation): make shadow extraction exact bounded and resumable

## 9. 阶段 E：接通独立的 P2 span-first 实体 treatment

### 9.1 端到端数据流

现有 SpanFirstEntityExtractor 若只在独立模块/测试中存在，不算完成。把 treatment 接到真实 snapshot pipeline：

    raw active revision Chunk
      -> span-first mention extraction
      -> structural/noise filtering
      -> type assertions
      -> exact alias retrieval
      -> indexed trigram/vector retrieval
      -> bounded reranker
      -> optional bounded LLM decision
      -> cannot-link enforcement
      -> versioned shadow assignment/canonical
      -> relation treatment endpoints

control 路径保持不变，直到 Gold gate 允许切换。

### 9.2 Shadow 必须独立

禁止重放 control mentions/assignments 后只做 exact alias。P2 shadow namespace 至少独立保存：

- mention text、exact span、chunk/revision；
- structural filter decision/reason；
- type assertions及来源；
- shadow canonical entities和 aliases；
- retrieval candidates、各阶段 score、rank、cap；
- final assignment/new canonical/uncertain；
- must-link、cannot-link 与 hard identifier 约束；
- embeddings/version；
- merge/split/undo event；
- provenance/model/config/code/policy hash；
- 下游 relation subject/object shadow endpoint。

正式 canonical、alias、mention、assignment、vector、projection 和 graph 必须零写入。

### 9.3 Identity 与 type 解耦

- canonical identity 不以当前预测 type 作为不可撤销分区。
- type 是有版本/来源/置信度的 assertion，可随新证据变化。
- 跨类型召回可进入候选，但 hard identifier/cannot-link 永远不能被相似度或 LLM 绕过。
- merge 是可审计、可逆事件；split/undo 后 projection、vector、source_count、relation endpoints 可以确定性重建。
- canonical name 不要求字面出现在每个 mention 的正文中；但发布关系端点必须能追溯到当前 active revision 的真实 mention span。保留 P0.1 的反幻觉约束，而不是拒绝合法 alias 归一。

### 9.4 有界检索与性能

- exact canonical/alias 命中不调用 embedding/reranker/LLM。
- alias 查询不能全项目 unnest。
- trigram/ANN 使用数据库索引；不把全项目 N 个实体加载到 Python 两两比较。
- 每个 mention 的 reranker pairs <= configured K，K 有硬上限并在 Gold 上校准。
- embedding 和 reranker 批处理，连接/HTTP 次数随 batch 数增长。
- source_count 只重算受影响实体/文档，不扫描全部历史 mentions。
- 默认向量维度必须有可用 HNSW；不支持维度显式 degraded/fail closed。

### 9.5 P2 测试

覆盖：

- alias-only mention 可归一到 canonical，同时 span 仍来自正文；
- 中英文、缩写、型号、版本、同名异义、OCR 噪声；
- cannot-link 和 hard identifier 冲突为零误合并；
- exact hit 零模型调用；
- candidate Recall@K 包含 Gold 而非只评最终命中；
- merge、split、undo、重启、重复运行；
- P2 结果进入 P1 treatment，端点来自同一 snapshot；
- control 表、正式向量和图零写入；
- 10k/100k canonical 的 EXPLAIN 走期望索引且候选有上限。

提交：

    fix(entity): run independent span-first resolution in shadow

## 10. 阶段 F：实现完整 policy snapshot orchestrator

### 10.1 不可变 manifest

已有 create/complete 数据库方法没有业务调用方时不算闭环。实现实际 service job 和 CLI，名称可按项目风格适配，例如：

    .venv/bin/python -m knowledge_forge.policy_snapshot build \
      --project-id TEST_PROJECT \
      --mode relation_only \
      --policy-set treatment-v1

    .venv/bin/python -m knowledge_forge.policy_snapshot build \
      --project-id TEST_PROJECT \
      --mode end_to_end \
      --policy-set treatment-v1

manifest 至少冻结：

- snapshot_id、mode、project namespace；
- document_id + active revision_id + content hash 的有序列表；
- control/treatment 完整 policy-set；
- entity/relation/verifier/model/embedding/reranker 名称和 digest；
- code commit/tree；
- 配置规范化 hash；
- prompt/schema/version；
- Gold manifest、gate thresholds、metric formulas 和 calibration universe version；
- created_at、created_by、input count；
- manifest hash。

创建后不可改输入；任何变更生成新 snapshot。

production/release snapshot 要求 tracked worktree clean 且 commit 已存在于 Git 历史。dirty tree 只能生成 pilot snapshot，并额外保存 dirty=true、tracked diff hash 和可能影响执行的 untracked code 清单。Gold、阈值和统计方法必须在首次查看 holdout 结果前进入 manifest hash；不能看完 holdout 再修改门槛。

### 10.2 可恢复 orchestrator

顺序：

1. validate manifest；
2. materialize control artifact；
3. relation_only 时装载冻结 entity spans/IDs；
4. end_to_end 时从原始 Chunk 跑 P2 treatment；
5. 在相同 snapshot endpoint 上跑 P1 treatment；
6. 等待所有 queue/outbox 达终态；
7. 写不可变 item artifacts 和 content hashes；
8. 计算 completeness；
9. 只有预期 item 集与实际 item 集精确相等且无 pending 才 finalize complete。

实现：

- DB-backed lease，支持多进程；
- heartbeat/expiry；
- bounded retry 和 terminal reason；
- idempotency；
- resume/status/cancel 命令；
- immutable artifact，重复写 hash 不同则失败；
- stale worker compare-and-set；
- 进程重启恢复；
- 不用一个长数据库事务包住整个 snapshot。

### 10.3 两类评估不可混淆

- relation_only：固定实体 spans/IDs，只衡量 P1 关系候选/分类/verifier。
- end_to_end：从原始 Chunk，经 P2 到 P1，衡量整个 treatment。

relation_only 通过不能授权 end_to_end 上线；后者实体变化后必须重新评关系。

测试中断每个阶段后 resume，证明：

- item 不重不漏；
- artifact hash 稳定；
- formal control/production 表零写入；
- snapshot 输入变动不会改旧 artifact；
- incomplete snapshot 无法进入 evaluate/release gate。

提交：

    feat(policy): orchestrate immutable resumable shadow snapshots

## 11. 阶段 G：让 P3 指标和 release gate 只能由 artifact 证明

### 11.1 Gold manifest

提供 pilot Gold 和 release Gold 两套等级：

pilot 可用于工程调试，但不能 production_approved。release Gold 必须：

- 非空 holdout；
- dev/holdout 无文档、Chunk 或内容 hash 重叠；
- 语料内容 hash 冻结；
- schema 和 policy 兼容范围明确；
- 每条标签记录来源；
- 至少两名独立 reviewer 或明确 adjudication；
- reviewer、reviewed_at、freeze_at 存在；
- synthetic/model-only 标签不得用于 release gate；
- 达到 README 声明的最小数据量。

建议 release 最小规模保持原计划：

- 200～300 个代表 Chunk，至少两个领域；
- >=500 entity mentions；
- >=150 must-links；
- >=150 cannot-links；
- >=200 positive relations；
- >=300 relation hard negatives。

如果实际未达到，只能 pilot=true，门禁必须拒绝生产激活。

### 11.2 在完整 universe 上计算实体聚类

修复 pairwise 与 B³ 只在 gold/prediction 交集计算的问题。预先冻结 span 对齐和公式，同时报告 detection-aware end-to-end clustering 与 matched-only 诊断指标：

- Gold mention 通过 document、revision/content hash、chunk、exact span 定位；
- missed Gold mention 在 predicted partition 使用唯一 missing sentinel/零贡献，并计入 recall；
- spurious prediction 在 augmented union 中使用唯一 Gold singleton 或等价明确 FP 规则；
- ambiguous overlap 不允许挑选最有利匹配；
- 多余 prediction 计 false positive/污染；
- unknown/missing 不能从分母删除；
- pairwise、B³、false merge、missed merge、duplicate canonical 使用预声明完整/增强 universe；
- mention span extraction 另算 exact/overlap/type 指标。

用可手算 fixture 锁定公式，并增加对抗测试：删除所有难例 prediction 后，指标必须下降，不能反而上升或不变。

### 11.3 严格关系和校准

严格 relation key 必须同时包含：

- document_id、revision/content hash、chunk_id；
- subject Gold mention/canonical ID；
- predicate；
- object Gold mention/canonical ID；
- direction；
- exact evidence_text；
- evidence start/end；
- policy/snapshot lineage。

只端点和 predicate 相同但证据坐标、方向或 Chunk 不同不能算 strict true positive。

shadow canonical ID 不能直接和 Gold canonical ID/名称比较。先用 (document, revision/content hash, chunk, exact span) 将 predicted subject/object mention 对齐到 Gold mention，再取得 Gold canonical endpoint。端点漏抽、多个 span 歧义、错误 assignment 必须按固定规则计 FN/FP，并分别归因 entity extraction、resolution 或 relation stage。

候选和最终指标分别计算：

- relation candidate Recall@budget；
- strict precision/recall/F1；
- evidence coverage/exactness；
- hard-negative false positives；
- missing/deferred/failed positives；
- NO_RELATION 质量。

Brier/ECE 使用 control/treatment 共用、在 manifest 中冻结的 calibration universe：全部 Gold positives + curated hard negatives/NO_RELATION，或另一套预先声明的固定集合。两侧每个 item 都必须有概率；未规划/失败/deferred 的正例按概率 0，未规划负例按 manifest 中预先声明的规则处理。各 planner candidate union 只能作为诊断，不能替代共同 universe。可靠性 bins 的边界和空 bin 规则固定。

### 11.4 Gate API

删除或封闭 evaluate_policy_snapshot_gate(snapshot_id, caller_supplied_metrics) 之类可注入汇总值的路径。release gate 只接受 evaluation_run_id：

1. evaluation_run 绑定 immutable snapshot artifact、Gold manifest 和 evaluator code/version；
2. evaluator 从 artifact 自己计算 metrics；
3. 保存完整计数、分母、item hashes 和报告 hash；
4. gate 重新校验 snapshot/Gold/artifact/evaluation hash、完整性和 release 等级；
5. 调用者不能提交任意 precision/F1 让 gate 通过。

不可变评估输入被篡改、缺少 item、hash 不同、evaluator 版本不允许、holdout 非 release 等均 fail closed。

Gold/artifact/evaluation 事实必须 append-only；普通应用运行角色无 UPDATE/DELETE 权限。evaluation 由受信 evaluator 从 item facts 生成，activation 角色只接收 evaluation_run_id 并重校验，必要时重算。reviewer 字符串只是元数据，release 还需受信 reviewer approval/签名或等价不可伪造批准。记录 holdout 解封和评估次数，避免反复调参后仍声称“冻结”。

### 11.5 统一 CLI 和门禁

升级 benchmarks/evaluate_kg_gold.py 或提供等价 CLI，一次输出：

- entity exact/overlap/type；
- entity candidate Recall@K；
- pairwise/B³；
- false/missed merge；
- relation candidate recall；
- strict relation P/R/F1；
- evidence exactness；
- Brier/ECE/reliability bins；
- control/treatment paired discordant counts；
- bootstrap 或 Wilson 区间；
- token、模型/DB/图调用、p50/p95；
- release gate 每一项 pass/fail/not-applicable。

硬安全门禁：

    invalid evidence publication = 0
    old/wrong policy publication = 0
    verification failed publication = 0
    cannot-link or hard identifier false merge = 0
    critical Gold relation loss = 0
    evidence exact locatability = 100%

有足够 release Gold 时：

    entity mention F1 不劣于 control 0.5 percentage point
    B3 F1 不劣于 control 0.5 percentage point
    relation recall/F1 不劣于 control 1.0 percentage point
    relation candidate recall >= 99%

还必须至少实现一个预先声明的实质收益；若质量无改善且成本更高，保持 shadow。非劣和实质收益应使用预声明的一侧 95% paired confidence interval 边界，而非只比较点估计；小样本则执行零新增关键错误和逐例审查。

测试包括 Gold/artifact/evaluation 每一层 hash tamper、空 holdout、缺 reviewer、交集作弊、缺 prediction、方向/证据错误、caller metric injection。

提交：

    fix(quality): derive release gates from immutable full-universe artifacts

## 12. 阶段 H：HugeGraph namespace、内容 manifest 和并发安全

### 12.1 稳定图 namespace

不要再只用 PostgreSQL 自增整数 project_id 作为图隔离边界。新增稳定 GRAPH_NAMESPACE_ID：

- UUID/ULID 或同等全局唯一、不可由测试 DB ID 碰撞的标识；
- 所有 vertex/edge logical ID、properties、索引、查询和 purge 均包含 namespace；
- project_id 可作为业务属性，但不能是唯一破坏性边界；
- graph instance/graph name + namespace 共同定义所有权；
- 旧图必须通过显式 migration/rebuild 迁移，不能静默兼容混写。

### 12.2 破坏性 guard

任何 purge/rebuild/failure-injection 前必须运行同一个 guard，并核验：

- 显式 RUN_HUGEGRAPH_INTEGRATION=1；
- destructive 操作另需 RUN_HUGEGRAPH_DESTRUCTIVE_TESTS=1；
- host、port、graph name、namespace 全部显式；
- 当前身份/版本 endpoint 可达；
- 图内存在与本次 run 匹配的一次性 sentinel；
- sentinel 包含随机 run_id、namespace、expires_at、purpose=test；
- destructive live test 的 (scheme, host, port, graph name) 必须与 production 物理图身份不同；仅 namespace 不同不算物理隔离；
- production 图身份未配置或无法证明不同时 fail closed；
- 禁止默认 graph 名 knowledge_forge；
- 禁止空 namespace、通配符、仅 project_id；
- 每次 mutation 都携带 guard context。

任一检查失败时，mutation HTTP 调用数必须为 0。固定生命周期：

    preflight(read-only, zero mutation)
      -> arm(只允许完全空的 disposable physical graph)
      -> execute
      -> cleanup

arm 生成的短期 token 绑定 endpoint、graph、namespace、run_id、config hash 和 expiry。execute 与 cleanup 均重新读取并验证 sentinel/guard；cleanup 只能删本次 namespace/sentinel，不能做全图清理。共享 physical graph 上一律禁止 schema drop、全图 purge 和 destructive live failure injection。

名为“非破坏性 HugeGraph integration”的测试必须是严格零 mutation 的只读 contract/guard 测试，并用 transport 计数断言。任何测试 vertex/edge 写入，即使最终清理，也必须进入 destructive marker、双开关和 sentinel 流程。

### 12.3 Graph content manifest

数量相等不足以 ready。构建确定性 expected manifest，至少覆盖：

- 每个 vertex label/edge label 数量；
- logical ID 的排序流式 digest；
- edge 的 subject logical ID、predicate、object logical ID、direction；
- namespace、project、graph generation、policy set；
- 必需 properties 的规范化 digest；
- relation assertion/projection lineage。

HugeGraph actual manifest 用稳定分页读取并生成相同 digest。不得把整图无界加载 Python，不得 O(n²) 比较；使用有界分页、外部排序/临时 PostgreSQL 表或流式分桶 hash。必须防分页重复/遗漏。

manifest 编码协议必须版本化并固定：canonical JSON、UTF-8、Unicode NFC、固定字段/类型、稳定浮点规则、长度无歧义编码、排序键和重复 logical ID 的拒绝语义。分页核对期间使用同一 generation/lease barrier 和可证明 snapshot/watermark；并发变化则本次验证失败重试。expected、writer-reported、actual 三份 manifest 必须在同一 generation、policy、namespace、lease token 下比较。

只有以下全真才能 CAS ready：

- expected manifest complete；
- writer reported manifest 匹配；
- actual HugeGraph manifest 匹配；
- 当前 generation/policy/namespace 仍等于 rebuild 开始时目标；
- 没有未完成 outbox/work item。

增加等量但错端点、反方向、错 predicate、错 label、混入旧 generation、漏一增一、分页重复等测试，全部保持 failed/stale。

### 12.4 跨进程 graph lease

现有进程内锁不够。增加 PostgreSQL-backed graph lease：

- namespace/project + generation 唯一；
- owner、heartbeat、expires_at、attempt；
- stale owner 不得写/ready；
- 新 generation 能 supersede 旧 generation；
- 单次 DB 事务很短，不持事务执行 HugeGraph HTTP；
- 每批写前/后校验 lease token；
- crash 后可回收，重试收敛。

同一项目并发 10 次文档完成应 coalesce 为稳定 generation 的一次完整 rebuild；若期间出现新 generation，最多再跑必要的新一代，旧代不能 ready。

提交：

    fix(graph): isolate namespaces and validate complete generation manifests

## 13. 阶段 I：版本化迁移、最小暴露和可复现 CI

### 13.1 版本化 migration

把启动时巨型 bootstrap DDL 演进为：

- schema_migrations(version, checksum, applied_at, duration, status)；
- 全局/数据库 advisory migration lock；
- 每个 migration 固定 checksum，已应用脚本改变则 fail；
- plan/status/apply CLI；
- additive、可重入的小步骤；
- 大 backfill 分批、可恢复、有进度；
- CREATE INDEX CONCURRENTLY 等不能处于事务内的步骤有显式状态；
- 启动默认只应用安全快速 migration，重 migration 由运维命令触发；
- 备份、恢复、回滚/forward-fix runbook。

审查并移除 bootstrap 中对旧 RAGFlow 列等直接 DROP。破坏性 cleanup 只能是独立、显式 opt-in 命令，先证明无引用并要求备份确认。HNSW/full backfill 不得每次启动重跑。

测试：

- 空库安装；
- 从 P0.2/P1/P2/P3 代表 schema clone upgrade；
- 连续 apply 两次幂等；
- checksum tamper 拒绝；
- 两个迁移进程并发；
- 中途失败后 resume；
- 旧数据、审计和 active revision 数量/内容 hash 不变；
- plan 模式零写入；
- backup/restore 后重新校验。

### 13.2 网络和凭据

- bare-host 默认监听 127.0.0.1，而不是 0.0.0.0；
- docker compose 暴露到宿主端口也默认绑定 127.0.0.1；
- dev 隔离模式的 0.0.0.0 需要显式危险开关并打印认证/网络警告；
- production/shared 模式若没有有效认证和显式运维批准，公网绑定必须启动失败，不能只警告；
- 容器内部监听地址与宿主机 published port 分开验证；容器内部可为网络通信监听 0.0.0.0，但宿主机默认仍只发布 loopback；
- 移除弱默认 PostgreSQL/HugeGraph 凭据，缺少 secret 时 fail fast；
- README 明确当前若没有认证/授权，禁止部署到共享 UAT、互联网或不可信网络；
- 不在本任务中草率自创不兼容 auth；若项目已有 auth 机制则补全测试。

审计事件可能包含正文/Prompt，增加默认 redaction、保留期限/容量配置和清理作业；release 测试只用合成/脱敏文档。

### 13.3 Docker 构建上下文

新增 .dockerignore，至少排除：

    .git
    .venv
    __pycache__
    .pytest_cache
    .env
    .env.*
    !.env.example
    KnowledgeFiles
    artifacts
    benchmarks/results
    logs
    dumps
    *.dump
    *.backup
    keys
    secrets
    *.pem
    *.key
    *.p12
    *.crt
    .coverage
    htmlcov
    Knowledge-Forge-Codex-*.md

若项目需要提交 .env.example，使用精确否定规则保留它，不得把其他 .env 重新纳入。测试在上述路径放置 canary 内容，并同时验证 Docker build context 和最终 image 中不存在 canary，不能只检查 .dockerignore 存在或 docker build 成功。

确认镜像中不包含 Git 历史、凭据、用户语料、数据库 dump、测试输出或任务文档。Dockerfile 使用固定基础镜像 digest 或明确版本，并以非 root 运行（若架构允许）。

### 13.4 CI

新增 .github/workflows/ci.yml，至少包含：

1. unit：
   - workflow permissions 最小化为 contents: read；
   - checkout persist-credentials: false；
   - actions 固定可信版本，安全策略允许时固定完整 SHA；
   - job timeout 和 concurrency cancellation；
   - PR 路径不读取 production/live secrets；
   - 固定 Python；
   - 可复现依赖安装；
   - pip check；
   - pytest 非 live markers；
   - py_compile；
   - diff/格式检查（只运行仓库实际配置的工具）。
2. postgres-integration：
   - 固定 PostgreSQL 16 和 pgvector 版本；
   - 运行 migration/revision/transaction/EXPLAIN tests；
   - 对 integration tests collected 数和 passed 数设置下限；
   - 如果全部 skipped，job 必须失败。
3. docker-build：
   - docker compose config；
   - 构建镜像；
   - 检查敏感路径不在 context/image；
   - 最小健康启动（可用 fake external services）。

不要让 CI 连接真实 Ollama/HugeGraph。依赖使用 constraints/lock 和 hash/固定版本；提交生成方法，避免每次安装漂移。

推送 main 前必须等待远程 CI 对精确当前 HEAD SHA 的所有 required jobs 成功。可用已登录的普通 gh CLI 或 GitHub API 只读核验；不能只看分支上“最近一次绿色”。无法独立核验精确 SHA 时，只推工作分支，不推 main，也不能报告 engineering_complete=true。

提交：

    chore(release): add migrations safe defaults and reproducible ci

## 14. 阶段 J：统一隔离 E2E、故障恢复和性能质量报告

### 14.1 单一 runner

实现 scripts/run_isolated_e2e.py 或等价工具：

    .venv/bin/python scripts/run_isolated_e2e.py --preflight-only
    .venv/bin/python scripts/run_isolated_e2e.py --execute

preflight-only 必须对 PostgreSQL、HugeGraph 和业务 API 全部零写入，并打印经过脱敏的：

- 规范化 PostgreSQL server/database/schema 身份；
- 是否为允许的 disposable database/schema，并与 production 配置硬比较；
- pgvector version；
- Ollama/embedding/reranker endpoints、model/digest/context；
- HugeGraph host/port/graph/namespace/identity；
- sentinel 和 destructive flags 状态；
- API bind/auth warning；
- Git commit/tree 和 policy/config hashes；
- 即将运行的场景与资源上限。

execute 只有 preflight 返回可机器校验 token 后才能运行；token 短期、绑定所有 endpoint/namespace/run_id，环境改变立即失效。

数据库名称或 TEST 环境变量本身不能证明安全。arm 阶段在权限受限的随机 schema 创建 PostgreSQL sentinel（run_id、identity、config hash、expiry）；execute/cleanup 每次重验。测试角色只能操作该 schema，不能 DROP database、其他 schema 或业务表。production 身份未知、身份相同、权限过宽且无法约束、sentinel 不匹配时 fail closed。增加 cleanup 越界测试，确认零条越界 SQL。

### 14.2 E2E 场景

只使用合成/脱敏 fixture，在隔离资源中完成：

- 上传 1、3、10 篇文档；
- 普通中英文、表格、dense、多句、重复句、URL/版本/小数；
- span-first P2 shadow、relation-only 和 end-to-end snapshots；
- 同一文档重入库三次；
- Chunk 中途失败和恢复；
- worker/service 重启；
- 两个并发 reingestion 乱序；
- delete、restore、再次 reingest；
- outbox/embedding/reranker/model timeout；
- HugeGraph 某批失败、数量相等但内容错、lease 过期和安全 retry；
- graph 409 -> rebuilding -> manifest exact -> ready；
- lexical/vector/search/list/statistics/ID 回取/graph 一致；
- control 与 shadow 正式表零污染；
- D=10 rebuild coalescing；
- final cleanup 只作用本次 test namespace/schema。

故障注入后必须证明：

- PostgreSQL 事务一致；
- 旧 active revision 可继续读；
- queue/lease 可恢复；
- 未验证关系为零发布；
- graph 不错误 ready；
- retry 最终收敛或明确 terminal failure；
- cleanup 可重复执行且不越界。

### 14.3 性能门禁

CI 硬门禁以操作数、增长率、token/payload、队列上限和 EXPLAIN 为主，避免跨机器绝对时间抖动：

- compact relation Prompt 的同一 context 只出现一次；
- 每个请求不超过已验证 context；
- exact entity hit 零 embedding/reranker/LLM；
- reranker pairs <= M × configured K；
- 模型/DB 往返随 batch 数增长；
- alias/trigram/ANN 查询不做项目全扫描；
- relation vector 在 LIMIT 前过滤；
- D=10 稳定批次不做 10 次完整 rebuild；
- cache/queue/lease/outbox 均有上限和生命周期；
- deferred 在 runner 结束时为 0 或明确 terminal，并计质量失败。

同一硬件 before/after release 报告目标：

- 普通语料 p95 不恶化超过 5%；
- dense p95 改善至少 20%；
- 候选 Prompt tokens 下降至少 40%；
- 实体阶段 DB/Embedding 往返下降至少 50%。

质量优先：若性能目标需要丢 Gold candidate、放宽 evidence、扩大 false merge 或长期积压才能达到，则门禁失败，保持 control。

提交：

    test(e2e): verify isolated recovery quality and performance gates

## 15. 必须实际执行的验证矩阵

命令需按最终真实文件名/marker 微调，但覆盖不得减少。每条命令在最终报告列出退出码、passed、skipped、failed 和耗时。

### 15.1 默认安全回归

    set -euo pipefail
    .venv/bin/python -m pytest -q -ra --strict-markers \
      -m "not integration and not live_model and not hugegraph_integration"
    .venv/bin/python -m py_compile knowledge_forge/*.py server.py
    .venv/bin/pip check
    git diff --check

如仓库配置 Ruff、Mypy、formatter 或前端检查，运行其全仓正式命令。若没有，不臆造工具；CI 新增工具时必须固定版本、加入配置并修到全仓通过。

### 15.2 Targeted blocker 回归

    .venv/bin/python -m pytest -q -ra --strict-markers \
      tests/test_document_content_revisions.py \
      tests/test_ollama_token_contract.py \
      tests/test_relation_shadow_recovery.py \
      tests/test_entity_shadow_independence.py \
      tests/test_policy_snapshot_orchestrator.py \
      tests/test_release_gate_artifacts.py \
      tests/test_hugegraph_safety_guard.py \
      tests/test_graph_manifest_validation.py \
      tests/test_migrations.py

如果按仓库风格改了测试文件名，最终报告提供映射。

### 15.3 Disposable PostgreSQL

必须使用临时 database 或随机 schema，绝不能指向业务库。优先提供仓库脚本自动创建、验证名称、执行、最后清理：

    POSTGRES_HOST=<TEST_HOST> \
    POSTGRES_PORT=<TEST_PORT> \
    POSTGRES_USER=<TEST_USER> \
    POSTGRES_PASSWORD=<TEST_PASSWORD> \
    POSTGRES_DB=postgres \
    POSTGRES_MAINTENANCE_DB=postgres \
    .venv/bin/python scripts/run_disposable_postgres_tests.py \
      -m integration -vv -ra --strict-markers

runner 必须保证：

- 测试 database/schema 名含随机 run_id 和允许前缀，但不把名称当唯一证明；
- 规范化 server/database/schema 身份与 production 硬比较；
- arm 后存在绑定 identity/config/run_id/expiry 的 sentinel，测试角色权限仅限随机 schema；
- server/version/pgvector preflight；
- 测试数量为零或全部 skipped 时非零退出；
- finally 清理只针对已验证的随机目标；
- cleanup 失败显式报告，不扩大删除范围。

至少运行 revision/FK/迁移/transaction/lease/outbox/EXPLAIN/Top-K 前置过滤测试。

### 15.4 真实模型 smoke

只在显式隔离 endpoint 和测试模型具备时：

    RUN_LIVE_MODEL_TESTS=1 \
    OLLAMA_BASE_URL=<ISOLATED_OLLAMA_URL> \
    RELATION_MODEL=<PINNED_MODEL> \
    ENTITY_MODEL=<PINNED_MODEL> \
    EMBEDDING_MODEL=<PINNED_MODEL> \
    RERANKER_MODEL=<PINNED_MODEL> \
    .venv/bin/python -m pytest -q -ra --strict-markers -m live_model

preflight 必须记录 model digest/context，正文只用合成 fixture。没有执行就报告未验证，不能以 fake 测试代替真实通过。

### 15.5 HugeGraph 四步门禁

第一步永远只读、零 mutation：

    .venv/bin/python scripts/hugegraph_test_guard.py preflight \
      --host <TEST_HOST> \
      --port <TEST_PORT> \
      --graph <NON_DEFAULT_TEST_GRAPH> \
      --namespace <UNIQUE_TEST_NAMESPACE>

第二步只读 integration，也必须零 mutation：

    RUN_HUGEGRAPH_INTEGRATION=1 \
    .venv/bin/python -m pytest -q -ra --strict-markers \
      -m "hugegraph_integration and not destructive"

第三步仅对完全空且物理隔离的 graph 显式 arm：

    .venv/bin/python scripts/hugegraph_test_guard.py arm \
      --host <TEST_HOST> \
      --port <TEST_PORT> \
      --graph <NON_DEFAULT_TEST_GRAPH> \
      --namespace <UNIQUE_TEST_NAMESPACE> \
      --run-id <UNIQUE_RUN_ID>

第四步只有 arm 返回并重新验证一次性 sentinel 后，才运行任何写入和破坏性失败注入：

    RUN_HUGEGRAPH_INTEGRATION=1 \
    RUN_HUGEGRAPH_DESTRUCTIVE_TESTS=1 \
    HUGEGRAPH_TEST_GRAPH=<NON_DEFAULT_TEST_GRAPH> \
    GRAPH_NAMESPACE_ID=<UNIQUE_TEST_NAMESPACE> \
    HUGEGRAPH_TEST_SENTINEL=<ONE_TIME_SENTINEL> \
    .venv/bin/python -m pytest -q -ra --strict-markers \
      -m "hugegraph_integration and destructive"

最后用同一 guard context 执行 cleanup；cleanup 前重新核验 sentinel，并断言只删除本次 namespace：

    .venv/bin/python scripts/hugegraph_test_guard.py cleanup \
      --graph <NON_DEFAULT_TEST_GRAPH> \
      --namespace <UNIQUE_TEST_NAMESPACE> \
      --run-id <UNIQUE_RUN_ID> \
      --sentinel <ONE_TIME_SENTINEL>

任何 placeholder、默认 knowledge_forge、空 namespace、无法证明隔离的 endpoint 都必须拒绝，且 mutation 调用数为 0。

### 15.6 Docker/CI 本地验证

    docker compose config
    docker build --progress=plain -t knowledge-forge:pre-live-test .
    .venv/bin/python scripts/run_isolated_e2e.py --preflight-only

具备所有隔离服务且 preflight 通过后：

    .venv/bin/python scripts/run_isolated_e2e.py --execute

### 15.7 Gold 和 release gate

实际 CLI 名称可适配，但必须展示等价步骤：

    .venv/bin/python benchmarks/evaluate_kg_gold.py \
      --snapshot-id <RELATION_ONLY_SNAPSHOT> \
      --gold-manifest tests/fixtures/kg_gold/pilot-manifest.json \
      --output artifacts/relation-only-evaluation.json

    .venv/bin/python benchmarks/evaluate_kg_gold.py \
      --snapshot-id <END_TO_END_SNAPSHOT> \
      --gold-manifest tests/fixtures/kg_gold/pilot-manifest.json \
      --output artifacts/end-to-end-evaluation.json

    .venv/bin/python -m knowledge_forge.release_gate \
      --evaluation-run-id <EVALUATION_RUN_ID>

pilot manifest 的预期是输出工程指标并明确 production gate 拒绝；只有真实 release manifest 才可能通过生产门禁。artifacts 默认不提交。

### 15.8 最终检查

    .venv/bin/python -m pytest -q -ra --strict-markers \
      -m "not live_model and not hugegraph_integration"
    .venv/bin/python -m py_compile knowledge_forge/*.py server.py
    .venv/bin/pip check
    git diff --check
    git status --short --branch
    git diff --stat origin/main...HEAD
    git diff --check origin/main...HEAD

## 16. 分阶段 commit、普通 push 和 main 对齐

建议提交顺序：

    test(readiness): reproduce pre-live release blockers
    fix(storage): make document reingestion revisioned and auditable
    fix(llm): separate context and output token budgets
    fix(relation): make shadow extraction exact bounded and resumable
    fix(entity): run independent span-first resolution in shadow
    feat(policy): orchestrate immutable resumable shadow snapshots
    fix(quality): derive release gates from immutable full-universe artifacts
    fix(graph): isolate namespaces and validate complete generation manifests
    chore(release): add migrations safe defaults and reproducible ci
    test(e2e): verify isolated recovery quality and performance gates
    docs: document isolated testing rollout recovery and release gates

实现允许合理合并/拆分，但每个 commit 必须单一目的、测试通过。每阶段：

    git status --short
    git diff
    git add -- <本阶段明确文件列表>
    git diff --cached --check
    git diff --cached --name-status
    git diff --cached
    git commit -m "<本阶段消息>"
    git push -u origin "$WORK_BRANCH"

后续阶段：

    git push origin "$WORK_BRANCH"

全部工程验证完成后：

    git push origin "$WORK_BRANCH"
    HEAD_SHA="$(git rev-parse HEAD)"
    gh auth status
    gh run list --branch "$WORK_BRANCH" --commit "$HEAD_SHA" --limit 20

等待该精确 HEAD_SHA 对应的 required workflows 完成，并逐个确认 conclusion=success；失败则查看日志、修复、重新测试、提交和 push。不得用其他 SHA 或“分支最近一次绿色”代替。若 gh 版本不支持上述过滤，使用只读 GitHub API 按 head_sha 核验。无法核验、workflow 未触发、required job 缺失或仍 pending 时，只保留工作分支，不能执行 main push。

精确 SHA 的 CI 全绿后再执行：

    git fetch --prune origin
    git rev-list --left-right --count HEAD...origin/main
    git log --oneline --left-right --decorate HEAD...origin/main
    git merge-base --is-ancestor origin/main HEAD

若 origin/main 已是 HEAD 祖先，且本文所有非 live 必须门禁通过：

    git push origin HEAD:main

如果 origin/main 有新提交：

1. 不 force、不 reset、不覆盖；
2. 审查远程新增 commit 和 diff；
3. 在已推送阶段分支上使用 git merge --no-edit origin/main，不 rebase 已推送历史；
4. 安全解决冲突；语义不明则停下报告；
5. 重跑所有受影响 targeted tests、完整非 live suite、PostgreSQL integration、编译和 diff check；
6. 普通 push 工作分支；
7. 再确认 origin/main 是 HEAD 祖先；
8. 普通 git push origin HEAD:main。

如果 branch protection 拒绝 main：

- 不绕过、不 force；
- 保留并推送工作分支；
- 报告远程分支名和 SHA；
- 给出创建 PR 所需信息，但不使用 GitHub App 代操作，除非用户之后明确要求。

推送后核验：

    git fetch --prune origin
    git rev-parse HEAD
    git rev-parse HEAD^{tree}
    git rev-parse origin/main
    git rev-parse origin/main^{tree}
    git ls-remote --heads origin main "$WORK_BRANCH"
    git status --short --branch
    git ls-files --others --exclude-standard

任务文档、凭据、真实正文、dump、模型日志、artifact 和 benchmark 临时输出不得提交。

## 17. 明确禁止

- 禁止给审计外键加 CASCADE 后继续硬删历史。
- 禁止把旧 verification/assertion 改 policy version 伪装成新策略通过。
- 禁止把 output max tokens 当 context budget。
- 禁止默认给 Ollama 发送 num_ctx 制造隐藏 runner 变化。
- 禁止用 str.find 的第一个 occurrence 猜重复 evidence 坐标。
- 禁止 candidates[:N]、截掉后部窗口或把 deferred 当 completed。
- 禁止用 control mention/assignment 重放冒充独立 P2 shadow。
- 禁止 shadow 写入正式 canonical/assertion/vector/search/graph。
- 禁止只在 gold/prediction 交集计算 pairwise/B3。
- 禁止让调用者提交自报 F1 直接通过 release gate。
- 禁止 model-only/synthetic label 冒充独立人工 release Gold。
- 禁止仅凭关系总数相同将 graph 标记 ready。
- 禁止只用数据库整数 project_id 作为 purge 边界。
- 禁止在默认 knowledge_forge 图、业务图或无法证明隔离的图上 purge。
- 禁止让 guard 失败后仍发任何 mutation。
- 禁止把全项目实体/关系/图加载到 Python 做无界或 O(n²) 比较。
- 禁止服务启动时重复大 backfill、HNSW 重建或破坏性 DROP。
- 禁止默认 0.0.0.0 暴露无认证服务到共享网络。
- 禁止提交 secret、.env、用户语料、日志、dump、Gold 私有正文或任务文档。
- 禁止删除失败测试、降低断言、随意 xfail/skip 或把 skipped 报 passed。
- 禁止 GitHub App、force-push、git reset --hard、git clean、git add . 或 git add -A。

## 18. README 和运维文档必须交付

至少说明：

- content revision 状态、重入库、失败恢复、删除/恢复；
- active revision 与历史审计查询差异；
- policy snapshot 的 relation_only/end_to_end、resume/status/cancel；
- Ollama context/max output 参数映射和 /api/show + /api/ps preflight；
- dense queue、lease、retry、terminal failure；
- P2 shadow namespace、merge/split/undo；
- Gold manifest、完整 universe 指标、evaluation_run 和 gate；
- GRAPH_NAMESPACE_ID、sentinel、只读/破坏性 HugeGraph 命令；
- graph manifest、lease、stale/rebuilding/ready/failed；
- migration plan/apply/status、备份恢复和 opt-in cleanup；
- loopback 默认、共享环境风险和 secret 配置；
- disposable PostgreSQL、live model、HugeGraph、Docker 和 E2E 命令；
- control 保持默认的条件、shadow/canary、回滚；
- 哪些测试真实执行，哪些需要用户提供隔离服务或人工 Gold。

## 19. 最终报告格式

最后用中文给出可核验报告：

1. 结论：engineering_complete、isolated_live_test_ready、production_approved 分别 true/false；
2. 每个 blocker 的根因、修复和测试证据；
3. 数据库 revision/外键/迁移和事务边界；
4. Ollama 实际 request contract；
5. P1/P2 是否真正从原始 Chunk 独立运行；
6. snapshot 完整性、artifact hash 和恢复证据；
7. entity/relation/calibration 指标的完整分母；
8. graph namespace/guard/manifest/lease 证据；
9. before/after 时间复杂度、操作数、token、DB/HTTP 调用、p50/p95；
10. 所有命令的 passed/skipped/failed/exit code；
11. PostgreSQL、Ollama、Embedding、reranker、HugeGraph 各自真实验证范围；
12. CI jobs 和 Docker 结果；
13. 未验证项、人工步骤和残余风险；
14. 每个 commit SHA、文件清单；
15. 开始/结束本地和 origin/main commit/tree；
16. 工作分支和普通 push 结果；
17. 明确确认未使用 GitHub App、force-push 或破坏性 Git 命令；
18. 未跟踪任务文档仍保留。

只有所有工程门禁真实通过，才可写 engineering_complete=true。只有隔离资源身份和安全 guard 已实际验证，才可写 isolated_live_test_ready=true。没有独立 release Gold 和 canary 证据时必须写 production_approved=false。

## 20. Codex 终止条件

不要因为单个阶段测试通过就提前停止。允许停止并报告阻塞的情况只有：

- origin/main 有无法安全判断的冲突；
- 工作区有与任务重叠且归属不明的用户修改；
- disposable PostgreSQL 权限不足且无法用安全替代环境完成真实 FK/锁测试；
- 真实模型或独立 HugeGraph 未提供；此时完成其余工程工作，并将对应 live 项明确标为未验证；
- production_approved 缺少独立人工 Gold/canary；此时保持 control，不能自行造标签。

除这些情况外，继续分析、修复、执行测试、自审、提交和普通推送，直至本文全部完成。
