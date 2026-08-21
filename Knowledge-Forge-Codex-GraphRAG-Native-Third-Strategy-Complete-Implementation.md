# Knowledge-Forge `graphrag_native` 第三策略完整实施任务书

> 版本：Microsoft GraphRAG 3.1.1 Native / Standard  
> 适用仓库：`https://github.com/zjx2024/Knowledge-Forge`  
> 目标分支：`fix/shadow-relation-quality-v2`  
> 参考实现：`https://github.com/microsoft/graphrag`  
> 已审计锚点：Knowledge-Forge `1ae83f531d2b18a1c9936585a8faa0db787740c1`；GraphRAG `v3.1.1` / `14a00ad88fc33cf2b52f4f113f25807556f8e25e`  
> 最终原则：执行时必须重新解析远端 HEAD；不得把上述锚点当成永远不变的目标 SHA

---

## 0. 如何把本任务交给 Codex

### 0.1 已有本地仓库

把本文件下载到一个明确路径，然后执行：

```bash
export KF_REPO="/absolute/path/to/Knowledge-Forge"
export KF_TASK="/absolute/path/to/Knowledge-Forge-Codex-GraphRAG-Native-Third-Strategy-Complete-Implementation.md"

git -C "$KF_REPO" fetch origin fix/shadow-relation-quality-v2
git -C "$KF_REPO" status --short --branch
git -C "$KF_REPO" rev-parse origin/fix/shadow-relation-quality-v2

set -o pipefail
export KF_CODEX_RUN_DIR
KF_CODEX_RUN_DIR="$(mktemp -d /tmp/kf-graphrag-native-codex.XXXXXXXX)"

codex exec \
  --cd "$KF_REPO" \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  --json \
  --output-last-message "$KF_CODEX_RUN_DIR/final-message.md" \
  - < "$KF_TASK" \
  | tee "$KF_CODEX_RUN_DIR/events.jsonl"
```

不要使用已经弃用的 `--full-auto`。本任务需要普通工作区写权限；外部服务、真实模型、高成本运行或受保护操作仍应由 Codex按权限边界处理。

如果 Codex 进程意外中断，在同一仓库继续：

```bash
export KF_SESSION_ID
KF_SESSION_ID="$(
  jq -r 'select(.type == "thread.started") | .thread_id' \
    "$KF_CODEX_RUN_DIR/events.jsonl" \
  | tail -n 1
)"
test -n "$KF_SESSION_ID"
test "$KF_SESSION_ID" != "null"

codex exec \
  --cd "$KF_REPO" \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  resume "$KF_SESSION_ID" \
  "继续执行 graphrag_native 第三策略任务书；先读取已有状态账本、git diff 和测试证据，从最后一个未通过门禁继续，不要重做已验证阶段。"
```

不要依赖 `resume --last`：它按当前工作目录选择最近 session，而启动命令并未要求 shell 先 `cd` 到仓库，可能恢复错误任务。若本机没有 `jq`，从 `events.jsonl` 的 `thread.started.thread_id` 手工复制 session ID。

也可以交互执行：

```bash
codex \
  --cd "$KF_REPO" \
  --sandbox workspace-write \
  --ask-for-approval on-request
```

进入后发送：

```text
完整阅读并严格执行
Knowledge-Forge-Codex-GraphRAG-Native-Third-Strategy-Complete-Implementation.md。
从 P0 开始连续推进；每阶段自检通过后自动进入下一阶段，直到达到当前权限和环境允许的最高完成状态。
```

### 0.2 尚未克隆仓库

```bash
export KF_PARENT="/absolute/path/for/source"
mkdir -p "$KF_PARENT"
git -C "$KF_PARENT" clone \
  --branch fix/shadow-relation-quality-v2 \
  --single-branch \
  https://github.com/zjx2024/Knowledge-Forge.git
```

然后按 0.1 执行。

### 0.3 不允许由启动命令隐式完成的操作

启动 Codex 不代表授权以下动作：

- 不得自动 `commit`、`push`、开 PR、合并或发布。
- 不得自动部署、切换项目默认策略或激活新策略。
- 不得重置、覆盖、暂存或删除用户已有修改。
- 不得修改或覆盖历史实验 bundle。
- 不得对非 disposable PostgreSQL 执行破坏性迁移试验。
- 不得在未授权时运行高成本真实模型重复实验。

---

## 1. 交给 Codex 的总指令

完整阅读本文件后再修改代码。本任务是在 Knowledge-Forge 中新增第三种独立、可选择的构建策略 `graphrag_native`，不是继续修补 `enhanced_shadow`，也不是把现有旧图转换成 GraphRAG BYOG 后只做社区检测。

从执行时实际解析出的远端目标分支 SHA 建立 `BASE_SHA`。读取所有适用的 `AGENTS.md`，检查工作树和用户已有修改，建立代码调用图与基线证据。然后从 P0 连续执行到 P11：

1. 每阶段先补 characterization test 或能证明当前缺口的失败测试。
2. 完成实现后运行本阶段 targeted tests。
3. 运行相关 disposable PostgreSQL、故障注入和前序回归。
4. 做代码自审、隔离审计和 artifact 自检。
5. 当前阶段硬门禁全部真实通过后，自动进入下一阶段，不等待逐阶段确认。
6. 如果 live 模型、Gold、外部服务或新权限缺失，继续完成不依赖该条件的后续工程工作；将对应门禁标为 `blocked` 或 `not_run`，绝不能伪报 `passed`。
7. 只有遇到本文件定义的暂停条件，或需要扩大授权、执行破坏性操作、处理用户冲突修改时，才停止并报告。

`graphrag_native` 必须：

- 从冻结的项目源文档和当前 active revision 开始；
- 使用 Microsoft GraphRAG 3.1.1 Standard workflow；
- 形成独立的 Documents、TextUnits、Entities、Relationships、Communities、Community Reports、Embeddings、LanceDB 与查询产物；
- 保留 GraphRAG 原生语义关系：

```text
source + target + natural-language description + weight + text_unit_ids
```

- 首期默认关闭 Claims/Covariates；
- 以不可变 Parquet + LanceDB bundle 作为算法事实源；
- 以 PostgreSQL 作为应用控制面；
- 在 HugeGraph 不可达时仍能完成 native 构建和查询。

`graphrag_native` 不得：

- 调用 Stable/Enhanced 的实体抽取、实体归一、canonical freeze；
- 调用旧 Schema、pair planner、距离过滤、pair classifier、proposer 或 verifier；
- 从旧 chunks、canonical graph、relation assertions 或 BYOG control graph 开始；
- 写旧正式实体、关系或 pgvector 表；
- 写 HugeGraph，或把 HugeGraph 作为 ready/query 条件；
- 失败后回退 Stable 或 Enhanced；
- 把自然语言关系强制伪装成旧固定谓词三元组；
- 用进程退出码 0、空结果、测试 skip、旧 artifact 或“关系更多”证明成功。

本任务只允许：

- 新增 GraphRAG-native 专属模块、表、测试、脚本、文档和 UI/API；
- 对共享策略派发、数据库、服务、配置、Web、Docker、CI 做必要的薄适配；
- 保持 Stable 和 Enhanced 的默认行为、已有数据、HugeGraph namespace、pgvector、查询结果、策略选择与 activation 不变。

---

## 2. 最终产品形态

完成后，应用中应存在三个明确策略：

```text
stable
enhanced_shadow
graphrag_native
```

三者的构建入口和事实源必须明确分开：

| 策略 | 构建事实源 | 图/向量服务 | 关系语义 |
|---|---|---|---|
| `stable` | 现有正式 PostgreSQL 实体关系 | 现有 pgvector / HugeGraph | 现有固定 Schema |
| `enhanced_shadow` | 现有 Shadow v2 隔离表 | 现有隔离图/审计链 | 固定谓词 hypothesis/fact |
| `graphrag_native` | 每次 run 的原生 Parquet + LanceDB | GraphRAG 原生查询；应用内视图 | source/target/description/weight |

GraphRAG-native 主链必须是：

```text
当前项目 active revisions
→ 结构感知 GraphRAG input Documents
→ GraphRAG 自己创建 TextUnits
→ 原生实体/关系联合抽取
→ finalize graph
→ 分层 Leiden communities
→ community reports
→ GraphRAG embeddings / LanceDB
→ Local / Global / DRIFT / Basic query
```

首期不增加 HugeGraph native 投影。Stable 和 Enhanced 仍需要 HugeGraph，因此服务可以保留，但 GraphRAG-native 对它的调用次数必须为 0。

---

## 3. 必须先重新核对的仓库事实

以下是审计锚点，不是允许跳过 P0 的静态真相：

1. 目标分支审计时 HEAD 为：

```text
1ae83f531d2b18a1c9936585a8faa0db787740c1
```

2. `requirements.txt` 已锁定：

```text
graphrag==3.1.1
```

3. CI 使用 Python 3.12，并已验证 GraphRAG 版本。
4. 当前策略常量只有 `stable` 和 `enhanced_shadow`。
5. `service.py` 中存在“Enhanced，否则 Stable”的二元隐式派发。
6. `migrations.py` 中已应用的多策略 migration 只允许两个策略。
7. `database.py` 有硬编码两策略初始化。
8. `web.py` 会把非 Stable 路线按 Enhanced/Hubble 处理。
9. 当前 `graphrag_community.py` 是旧正式 control graph → BYOG → communities，不是原生 GraphRAG index。
10. 当前 `community_reports.py`、`global_search.py` 读取旧 PostgreSQL/canonical 图，不是 native bundle。
11. 当前整体 health 可能被 HugeGraph 不可达拖成失败。
12. 现有 migration 序号审计时到 `0021`；执行时必须选择实际下一个空闲版本，禁止假定仍然是 `0022`。

如果远端 HEAD 已前进：

- 固定新的 `BASE_SHA`；
- 审查 `1ae83f5..BASE_SHA`；
- 更新符号和 migration 序号；
- 保留本任务的架构不变量；
- 不得自动 reset/rebase 到旧 SHA。

---

## 4. GraphRAG 3.1.1 原生契约

### 4.1 固定版本和运行边界

GraphRAG 参考版本：

```text
tag: v3.1.1
commit: 14a00ad88fc33cf2b52f4f113f25807556f8e25e
Python: >=3.11,<3.14
```

必须通过实际安装包自检，不凭文档记忆调用内部 API：

```bash
"$KF_PYTHON" -c \
  "from importlib.metadata import version; print(version('graphrag')); assert version('graphrag') == '3.1.1'"

"$KF_PYTHON" -m graphrag --help
"$KF_PYTHON" -m graphrag init --help
"$KF_PYTHON" -m graphrag index --help
"$KF_PYTHON" -m graphrag query --help
"$KF_PYTHON" -m graphrag prompt-tune --help
```

索引必须由隔离子进程运行。主 Web worker 不得在全局进程中 import 并长期持有 GraphRAG runtime，因为 GraphRAG API/config 加载会修改运行目录或全局状态。

查询如果只调用 CLI，无法稳定获得结构化 context；因此实现一个隔离 query worker 子进程，在固定 GraphRAG 3.1.1 adapter 中调用版本锁定的 Python API，并以受版本控制的 JSON 协议与主服务通讯。该 API 官方标注为仍在开发且不保证向后兼容，因此 worker 启动时必须断言函数签名和返回契约，GraphRAG 升级时重新验收。3.1.1 API 提供 response 和 context data，但不提供统一的规范化 citation 对象；Knowledge-Forge 必须自行定义 context serializer，并把“选入上下文的来源”与“答案文本中可解析、受支持的引用”分开。

### 4.2 Standard workflow

生产配置不得提供自定义 `workflows:` 覆盖。完整 Standard workflow 必须包含并审计以下十步：

```text
1. load_input_documents
2. create_base_text_units
3. create_final_documents
4. extract_graph
5. finalize_graph
6. extract_covariates
7. create_communities
8. create_final_text_units
9. create_community_reports
10. generate_text_embeddings
```

即使首期关闭 Claims/Covariates，也必须记录第 6 步的合法关闭/空产物状态，不能让 workflow ledger 缺项。

### 4.3 官方命令边界

只能在全新的 staging run root 中初始化：

```bash
"$GRAPHRAG_NATIVE_PYTHON" -m graphrag init \
  --root "$RUN_ROOT" \
  --model "$CHAT_MODEL" \
  --embedding "$EMBED_MODEL"
```

配置解析 dry-run：

```bash
"$GRAPHRAG_NATIVE_PYTHON" -m graphrag index \
  --root "$RUN_ROOT" \
  --method standard \
  --dry-run \
  --skip-validation \
  --verbose
```

模型连通性 dry-run：

```bash
"$GRAPHRAG_NATIVE_PYTHON" -m graphrag index \
  --root "$RUN_ROOT" \
  --method standard \
  --dry-run \
  --verbose
```

真实索引：

```bash
"$GRAPHRAG_NATIVE_PYTHON" -m graphrag index \
  --root "$RUN_ROOT" \
  --method standard \
  --verbose
```

索引完成后的 CLI 诊断查询：

```bash
"$GRAPHRAG_NATIVE_PYTHON" -m graphrag query \
  --root "$RUN_ROOT" \
  --method local \
  "$QUESTION"

"$GRAPHRAG_NATIVE_PYTHON" -m graphrag query \
  --root "$RUN_ROOT" \
  --method global \
  "$QUESTION"

"$GRAPHRAG_NATIVE_PYTHON" -m graphrag query \
  --root "$RUN_ROOT" \
  --method drift \
  "$QUESTION"

"$GRAPHRAG_NATIVE_PYTHON" -m graphrag query \
  --root "$RUN_ROOT" \
  --method basic \
  "$QUESTION"
```

CLI 查询只用于 operator smoke；应用集成仍使用隔离 query worker，因为 UI/API 需要结构化 context 和引用，而不是仅获得终端打印的答案。

规则：

- `--skip-validation` 只允许用于纯配置解析。
- 生产索引不得使用 `--skip-validation`。
- 带 `--skip-validation` 的 dry-run 只验证配置可以加载。
- 不带 `--skip-validation` 的 dry-run 会对每个 completion model 发出一次 Hello World，并对每个 embedding model 发出一次测试调用，因此可以证明最小连通性。
- GraphRAG 会根据测试 embedding 在当前进程内同步 vector dimension，但 dry-run 退出后不会把该覆盖写回 `settings.yaml`；Knowledge-Forge doctor/settings builder 必须显式探测并持久化真实维度。
- 连通性 dry-run 仍不证明 extraction tuple 或 community-report JSON 合法。
- 必须另跑固定 smoke corpus。
- 不使用 `graphrag update` 更新 immutable project revision。3.1.1 的 update 不能可靠识别同标题内容修改和删除。
- 每次 source/model/prompt/config 改变都创建新 run root 并全量 index。

可选 Prompt Tune 命令：

```bash
"$GRAPHRAG_NATIVE_PYTHON" -m graphrag prompt-tune \
  --root "$RUN_ROOT" \
  --domain "industrial minerals, alloys, mines, production, reserves and supply chains" \
  --selection-method random \
  --limit 15 \
  --language Chinese \
  --max-tokens 2000 \
  --chunk-size 600 \
  --overlap 100 \
  --min-examples-required 3 \
  --discover-entity-types \
  --output "$RUN_ROOT/prompts"
```

Prompt Tune 产物应包括：

```text
extract_graph.txt
summarize_descriptions.txt
community_report_graph.txt
```

Prompt Tune 是显式 profile 生成步骤，不得在每次生产构建中隐式重跑。最终使用的 Prompt 内容和 SHA-256 必须冻结。

### 4.4 原生输出，不做旧 Schema 强制映射

GraphRAG relationship extraction 语义近似：

```text
("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>STRENGTH)
```

最终关系核心字段：

```text
source
target
description
weight
text_unit_ids
```

这些字段不是旧式 `(subject, predicate, object)`。UI、API、导出和测试都必须称为“语义关系”，不得伪造 `produces/uses/located_in`。

GraphRAG 聚类会把关系用于加权无向图社区检测；不能用旧方向准确率作为 native 社区主门禁。

四类查询的最低输入依赖必须分别验证：

| Query | Parquet/结构依赖 | 向量依赖 |
|---|---|---|
| Global | entities、communities、community reports | 无 |
| Local | entities、communities、community reports、relationships、TextUnits；covariates 可选 | entity descriptions |
| DRIFT | entities、relationships、TextUnits、communities/reports | entity descriptions、community full content |
| Basic | TextUnits | TextUnit text |

### 4.5 Claims/Covariates 首期关闭

GraphRAG 3.1.1 的 Claims 路径存在两个已知版本限制：

- 实体类型使用默认集合，不能可靠继承领域化实体类型。
- gleaning 结果在当前实现中可能未进入最终返回。

因此首期配置：

```yaml
extract_claims:
  enabled: false
```

仍保留空 `covariates` 契约和明确状态。不得：

- patch 上游 GraphRAG 后仍把结果称为 unmodified 3.1.1 native；
- 用旧 Knowledge-Forge claim/schema 链补齐；
- 把未运行的 claims 标为通过。

后续若用户另立任务启用 Claims，应作为显式实验 profile、独立 Gold 和独立版本说明处理。

---

## 5. 可复用、薄适配与禁用边界

### 5.1 可复用

- `projects`、`documents`、源文件。
- `document_content_revisions`、active revision、source hash。
- Docling/OCR/图片/表格/列表的结构解析产物。
- `document_structure()` 读取已保存 parser blocks 的能力。
- PostgreSQL jobs、日志、lease、heartbeat、CAS、恢复和取消设计模式。
- `KnowledgeFiles` 持久卷。
- 当前项目选择的 completion/embedding 模型名和 provider endpoint。
- migration runner 与 disposable PostgreSQL 测试设施。
- 现有 GraphRAG 代码中的进程隔离、失败分类和 CAS 设计思路。
- 已锁定的 GraphRAG/LanceDB/PyArrow 依赖。

### 5.2 只允许薄适配

- `knowledge_forge/construction_strategies.py`
- `knowledge_forge/service.py`
- `knowledge_forge/database.py`
- `knowledge_forge/migrations.py`
- `knowledge_forge/config.py`
- `knowledge_forge/web.py`
- `.env.example`
- `.env.docker.example`
- `.gitignore`
- `Dockerfile`
- `docker-compose.yml`
- `.github/workflows/ci.yml`
- `README.md`
- `AGENTS.md`

薄适配必须通过旧策略 characterization tests 证明默认语义不变。

### 5.3 可借鉴、不得作为 native 输入

- `knowledge_forge/graphrag_community.py`
- `knowledge_forge/community_reports.py`
- `knowledge_forge/global_search.py`
- `knowledge_forge/shadow_context.py`

可以抽取纯结构映射、进程治理或错误分类的通用小组件，但：

- 不得调用旧 control-BYOG snapshot；
- 不得读取旧 canonical graph；
- 不得把旧 PostgreSQL community/report 表当 native output；
- 不得保留 legacy chunk fallback；
- 不得让 native 调用带 `shadow` 语义的实体/关系处理。

### 5.4 native 主链调用即失败的入口

隔离测试必须 monkeypatch 下列入口或其实际等价符号为“调用即抛错”：

```text
Stable _ingest_document extraction path
Enhanced _ingest_enhanced_strategy
span_entities / entity extraction
EntityResolver / canonical resolution
entity_shadow_*
relation_shadow_*
Schema overlay / Schema evolution relation discovery
pair planner / pair budget / distance filter
relation classifier / proposer / verifier
canonical_entities writes
entity_mentions writes
relation_assertions writes
canonical_relation_projections writes
VectorStore / pgvector native writes
HugeGraphStore native reads/writes
Database.export_graphrag_control_snapshot
old graphrag_community_runs/report/global-search inputs
```

门禁：

```text
graphrag_native → legacy extraction/resolution/relation calls = 0
graphrag_native → legacy formal entity/relation/vector writes = 0
graphrag_native → HugeGraph calls = 0
stable/enhanced → native-only calls = 0
unknown strategy → explicit error
```

---

## 6. 建议新增模块和职责

按仓库既有风格调整命名，但职责不得混合：

```text
knowledge_forge/graphrag_native_inputs.py
knowledge_forge/graphrag_native_config.py
knowledge_forge/graphrag_native_runner.py
knowledge_forge/graphrag_native_artifacts.py
knowledge_forge/graphrag_native_query.py
knowledge_forge/graphrag_native_query_worker.py

scripts/validate_graphrag_native_bundle.py
scripts/check_graphrag_native_freshness.py
scripts/export_graphrag_native_bundle.py
scripts/run_graphrag_native_pilot.py
scripts/graphrag_native_doctor.py

docs/graphrag-native-strategy.md
```

建议主要对象：

```text
GraphRAGNativeInputBuilder
GraphRAGNativeSettingsBuilder
GraphRAGNativeRunner
GraphRAGNativeBundleValidator
GraphRAGNativeQueryService
GraphRAGNativeExecutor
```

建议测试文件：

```text
tests/test_graphrag_native_strategy.py
tests/test_graphrag_native_migrations.py
tests/test_graphrag_native_inputs.py
tests/test_graphrag_native_config.py
tests/test_graphrag_native_runner.py
tests/test_graphrag_native_artifacts.py
tests/test_graphrag_native_service.py
tests/test_graphrag_native_isolation.py
tests/test_graphrag_native_query.py
tests/test_graphrag_native_web.py
tests/test_graphrag_native_postgres.py
tests/test_graphrag_native_live.py
```

禁止创建一个同时负责输入转换、数据库状态、GraphRAG 运行、查询和 UI 的巨型模块。

---

## 7. 控制面数据库设计

不得把 GraphRAG 原生 entities/relationships 全量复制进旧正式表。PostgreSQL只保存控制面与审计元数据。

至少建立以下逻辑表；实际列类型、命名和 FK 应匹配仓库约定：

### 7.1 `graphrag_native_runs`

至少包含：

```text
run_id
project_id
generation
state
base_sha
worktree_diff_hash
code_tree_hash
source_snapshot_hash
settings_hash
prompt_profile_hash
completion_model_id
completion_model_digest
embedding_model_id
embedding_model_digest
embedding_dimension
graphrag_version
python_version
staging_relative_path
bundle_relative_path
validator_version
validation_report_hash
created_at
started_at
heartbeat_at
execution_completed_at
validated_at
ready_at
failed_at
failure_class
failure_message
lease_owner
lease_expires_at
superseded_by
```

### 7.2 `graphrag_native_run_documents`

至少包含：

```text
run_id
project_id
document_id
revision_id
source_hash
structure_hash
input_document_count
input_manifest_hash
ordinal
```

必须用复合约束或事务校验阻止跨 project/document/revision 拼接。

### 7.3 `graphrag_native_artifacts`

至少包含：

```text
run_id
logical_name
relative_path
sha256
byte_size
media_type
schema_hash
row_count
created_at
```

路径必须：

- 相对 native root；
- 规范化后仍位于 run root；
- 不允许 `..`、绝对路径、symlink 逃逸；
- 不允许两个逻辑 artifact 指向同一可变文件。

### 7.4 `graphrag_native_project_state`

至少包含：

```text
project_id
active_run_id
active_generation
desired_source_snapshot_hash
updated_at
```

只有 independent validator 成功、source 仍 current 的 run 才能 CAS 成 active。

新 run 失败时，旧 ready/active run 保持可读。

这里的 `active_run_id` 只表示 native 策略内部当前可查询的 ready bundle，不等于把项目总 construction strategy 激活为 `graphrag_native`。项目总策略指针必须独立保存和授权；本任务可建立/测试 native active run，但不得自动改变项目总策略。

### 7.5 `graphrag_native_query_runs`

至少包含：

```text
query_run_id
project_id
run_id
query_mode
question_hash
status
model_id
started_at
completed_at
latency_ms
token_usage
context_reference_count
failure_class
```

默认不在普通日志中保存完整敏感问题和答案；按现有隐私策略决定是否加密或受控保存。

### 7.6 run 状态机

必须明确区分：

```text
PENDING
PREPARING
RUNNING
EXECUTION_COMPLETE
COMPLETED_WITH_FAILURES
VALIDATING
READY
EMPTY_INPUT
NO_GRAPH_UNSUPPORTED
FAILED
CANCELLED
STALE
SUPERSEDED
```

关键语义：

- `EXECUTION_COMPLETE` 只表示子进程结束，绝不等于 ready。
- 进程 0 退出但日志/产物有静默失败时进入 `COMPLETED_WITH_FAILURES` 或 `FAILED`。
- 零输入在调用 GraphRAG 前进入非 READY 的 `EMPTY_INPUT`，旧 active run 保持可读。
- 非空输入但零关系/零可聚类图在未修改的 3.1.1 Standard 中属于不支持的 `NO_GRAPH_UNSUPPORTED` 或 `FAILED`，不能合成空表后声称 native success。
- source/model/config/prompt/code 改变使旧候选 `STALE`。
- 晚到 worker 不能覆盖新 generation。
- `READY` 不得从“没有 pending row”推导。

---

## 8. 文件系统、输入和 artifact 契约

### 8.1 不可变运行目录

```text
KnowledgeFiles/graphrag_native/
  .staging/
    <run_id>-<attempt_id>/
  project-<project_id>/
    runs/
      <run_id>/
```

每个 run 至少包含：

```text
identity/
input/
prompts/
settings.yaml
settings.redacted.yaml
output/
output/lancedb/
cache/
logs/
kf-manifest/
```

规则：

- staging 必须为本次 attempt 新建的空目录。
- output/cache 不得复用历史 run。
- 失败重试使用新 attempt 目录，不原地覆盖。
- 完整验证后，使用同一文件系统内原子 rename 完成 finalize。
- DB active pointer 在 finalize 和独立验证完成后 CAS。
- 不自动删除失败或旧 ready run。
- retention 只提供 dry-run/候选列表；实际删除需另行授权。

### 8.2 GraphRAG 输入

使用 GraphRAG 3.1.1 官方支持的结构化 Parquet input，而不是旧 chunks 或 BYOG。

从每个项目当前非删除文档的 active revision 和持久化 parser blocks 构造确定性 Documents：

- 普通段落按 heading/section 分组，不机械地每 block 一个 document。
- 表格按行形成输入单元，每行重复必要列头、多级表头、表名、单位和年份。
- 列表保留 list group 和 item 边界，禁止把相邻国家/条目线性串接。
- 图片/OCR 内容保留页面和结构来源。
- 每个输入单元有稳定、唯一、不跨项目的 opaque ID。
- title 必须稳定且唯一；人类标题另存 metadata。
- 文本中加入最小、可识别的 source identity，避免两个文档完全相同文本产生 TextUnit ID 碰撞。
- extraction prompt 明确 source metadata 不是实体；加入污染测试。

Sidecar/source map 至少保存：

```text
project_id
document_id
revision_id
source_hash
input_document_id
input_title
block_ids
page
heading_path
table_id
row
column
header_path
list_id
list_item
source_start
source_end
rendered_text_start
rendered_text_end
```

GraphRAG provenance 只能承诺到 TextUnit/source-map 级。无法证明精确字符 span 时必须明确标注，不能伪装为旧系统的精确 evidence。

如果任何 active document 没有可用结构块：

```text
fail closed
```

不得回退 legacy chunks。

### 8.3 原生输出表

独立 validator 必须按 GraphRAG 3.1.1 实际 schema 校验至少：

```text
documents.parquet
text_units.parquet
entities.parquet
relationships.parquet
communities.parquet
community_reports.parquet
```

固定版本下至少核对下列官方字段；若实际安装包的 schema 定义与此清单不同，必须先以 tag source 和运行时 schema 证明差异，不能静默忽略：

| Table | 核心字段 |
|---|---|
| documents | id、human_readable_id、title、text、text_unit_ids、creation_date、raw_data |
| text_units | id、human_readable_id、text、n_tokens、document_id、entity_ids、relationship_ids、covariate_ids |
| entities | id、human_readable_id、title、type、description、text_unit_ids、frequency、degree |
| relationships | id、human_readable_id、source、target、description、weight、combined_degree、text_unit_ids |
| communities | id、human_readable_id、community、level、parent、children、title、entity_ids、relationship_ids、text_unit_ids、period、size |
| community_reports | id、human_readable_id、community、level、parent、children、title、summary、full_content、rank、rating_explanation、findings、full_content_json、period、size |

Claims 关闭时，`covariates` 应是明确关闭或合法空表状态。

最终 settings 必须打开 GraphRAG 原生审计快照，并保存：

```text
raw_entities.parquet
raw_relationships.parquet
graph.graphml（审计快照，不是事实源）
stats.json 或其他实际由 3.1.1 生成的可选文件（存在时）
kf-manifest/run-context.json（Knowledge-Forge 自建、版本化 schema）
LanceDB 三类表
完整 logs
run-local cache manifest
```

GraphML 只能用于调试和可视化，不能成为 canonical bundle。

`raw_entities.parquet` / `raw_relationships.parquet` 也不是模型原始响应：它们已经过 delimiter tuple 解析、聚合和 orphan 过滤，只是 description summarization 前的 GraphRAG 快照。不得用它们声称能够审计所有 malformed 或被静默跳过的模型 record。

GraphRAG 3.1.1 会对字段数或 delimiter 不合法的单条 extraction record 静默 `continue`。因此：

- 全部畸形并最终形成零图时，应进入非 READY 失败状态。
- 同一响应中部分合法、部分畸形时，原生 index 可能成功；Parquet validator 无法从解析后结果证明被丢弃 record 的数量。
- 必须用固定 smoke、Gold、模型调用聚合审计和质量门禁发现这一风险。
- 若实现原始响应捕获，只保存受控、脱敏、访问受限的 ledger，并说明 instrumentation 方式。
- 若改写 GraphRAG parser 使每条畸形都硬失败，必须声明为 Knowledge-Forge compatibility patch/profile，不能称为未修改的 3.1.1 native extractor。

---

## 9. 配置与模型契约

### 9.1 配置生成

不得复制旧 GraphRAG 版本的 YAML。每次以实际安装的 3.1.1 `graphrag init` 产物和当前 config model 为基准生成 typed settings。

3.1.1 顶层配置应使用当前键，例如：

```text
completion_models
embedding_models
input_storage
output_storage
update_output_storage
table_provider
```

不得使用旧版示例的 `models` / `output` 结构。

结构化 Parquet 输入必须显式映射稳定列；否则 3.1.1 会忽略自定义 ID/title，退回文本 hash 和“文件名 + row index”：

```yaml
input:
  type: parquet
  file_pattern: 'documents\.parquet$'
  id_column: id
  title_column: title
  text_column: text
input_storage:
  type: file
  base_dir: input
chunking:
  prepend_metadata:
    - id
```

`input_storage.base_dir` 中只放 GraphRAG input documents；source map 和其他 Parquet sidecar 放在该目录外。`chunking.prepend_metadata` 必须把稳定唯一字段加入每个 chunk，而不只是文档第一段，从而降低跨文档相同 chunk 的 TextUnit ID 碰撞。即便如此，同一 document 内完全相同 chunk 仍可能碰撞，因为 3.1.1 ID 不含 chunk ordinal；检测到时必须失败并重构输入单元，不能静默去重。

最终生产 settings：

- 不得包含自定义 `workflows:`；
- `cluster_graph.use_lcc: false`，覆盖所有连通分量；
- 固定 seed，建议使用官方默认 `3735928559`；
- 单 GPU 初始 `concurrent_requests: 1`；
- 不依赖 `drift_search.concurrency`：3.1.1 虽定义该字段，但 query 路径未实际读取；
- 单 GPU 首期设置 `drift_search.primer_folds: 1` 与 `drift_search.drift_k_followups: 1`；
- query worker/provider 层再用受测试的 semaphore 把真实 completion/embedding 最大并发限制为 1，并记录观测最大并发；
- cache 只在当前 run 内使用；
- 明确禁用 Claims；
- 明确每个 embedding target；
- 在 `vector_store.index_schema` 中显式固定三类 logical target 的 `index_name`，或由所有 consumer 从 resolved settings 读取；不得在 validator/query worker 另行硬编码与 settings 无关的物理表名；
- 明确启用 raw graph 与 GraphML snapshots；
- 保存 redacted settings 和完整 hash；
- API key 使用环境引用，不落盘、不进日志。

对应配置至少包含：

```yaml
snapshots:
  raw_graph: true
  graphml: true
  embeddings: false
```

首期以 LanceDB 为 embedding 事实源，因此 embedding Parquet snapshot 默认可关闭；若后续开启，必须加入 manifest 和 validator，不能混用两个来源。

### 9.2 Ollama/provider

通过实际 smoke 选择一种受支持配置：

```text
model_provider: ollama
api_base: http://<host>:11434
```

或 OpenAI-compatible：

```text
model_provider: openai
api_base: http://<host>:11434/v1
```

GraphRAG 3.1.1 默认 LiteLLM adapter 在 `auth_method=api_key` 且不是 Azure managed identity 时要求非空 `api_key`；Ollama/OpenAI-compatible 本地 endpoint 可通过环境提供非空 dummy key。任何 key 都只能放环境变量，不写 artifact。若改用 custom provider/auth，必须显式标记为兼容适配，不能称为零改动默认配置。

不得同时配置两个 endpoint 后在失败时静默切换。被选 provider、base URL 的非敏感部分、模型 digest 和 adapter profile 必须进入 manifest。

### 9.3 embedding 维度

不得接受 GraphRAG 默认 3072 作为假设。必须：

1. 对实际 embedding 模型发一次 probe。
2. 校验返回向量非空、数值有限、维度稳定。
3. 把实际维度写入最终 settings 的 `vector_store.vector_size`。
4. 同步写入 `vector_store.index_schema.<embedding>.vector_size`；当前启用的三类 index schema 必须全部一致。
5. 重新 `load_config`，运行不带 `--skip-validation` 的连通性 dry-run，再运行 live index。
6. 在 LanceDB validator 中逐表重算维度一致性。

例如 BGE-M3 常见维度可能是 1024，但禁止硬编码“因为模型名看起来是 BGE-M3 就等于 1024”。

### 9.4 cache 风险

GraphRAG 3.1.1 cache key 不能覆盖 `api_base`、auth、API version 等 endpoint 配置变化；provider/model 通常会进入模型标识，但仍不足以替代完整运行指纹。因此：

- cache 必须 run-local；
- provider/model/config/prompt 任一变化都创建新 run；
- fresh live E2E 必须使用空 cache；
- live model call attempts 必须非零；
- 旧 cache 不得复制到新 run。

---

## 10. 独立 bundle validator 的硬门禁

验收必须拆成两个不同层次：

1. `scripts/validate_graphrag_native_bundle.py` 是 offline integrity validator。它不读取数据库 summary，从 bundle 文件重算内部身份、输入快照、输出和引用；只传 `--root` 时不能声称知道项目当前 active revisions 是否变化。
2. `scripts/check_graphrag_native_freshness.py` 或同职责 service gate 是 online freshness validator。它从 PostgreSQL 重新计算项目当前 document/revision/structure snapshot，并核对当前 desired model/config/prompt profile/generation，与已通过 offline validator 的 manifest 比较。

两者都必须以非零退出/明确错误表示失败。只有 offline integrity 和 online freshness 同时通过，且 generation 未变化，才允许 CAS active pointer。

### 10.1 身份与 freshness

- GraphRAG 版本恰为 3.1.1。
- `BASE_SHA + WORKTREE_DIFF_HASH + CODE_TREE_HASH` 可重算。
- source document/revision/structure hash 与 input manifest 一致。
- settings、Prompt、模型、embedding dimension 和 runtime environment hash 一致。
- offline validator 只证明 bundle 内记录的 source snapshot 自洽。
- online freshness gate 必须证明当前项目 snapshot 与 bundle 完全相等；当前 source 改变后，旧 run 不得通过 activation/query freshness。

### 10.2 文件与 schema

- manifest 中每个文件存在、普通文件、无 symlink 逃逸。
- 路径规范化后位于 run root。
- bytes、SHA-256、media type、schema hash 和 row count 相符。
- 不允许 manifest 外的关键 output 偷偷参与查询。
- Parquet 可完整读取，必需列和类型合法。
- 对官方 final tables 校验 `id` 唯一。
- 3.1.1 的 community report ID 由 `full_content` 派生；两个社区生成完全相同文本时可能碰撞。Knowledge-Forge 将 final ID 唯一作为更严格的 READY 门禁：检测到时失败/重跑或拒绝，不得静默改写官方 report ID，也不得称为上游原生保证。
- `raw_entities.parquet` 没有 final `id`，按 `(title, type)` 复核 3.1.1 聚合语义；`raw_relationships.parquet` 按 `(source, target)` 复核。
- source-map 等 Knowledge-Forge sidecar 按各自版本化 schema/复合键校验，不能套用 final-table ID 规则。

### 10.3 GraphRAG 数据守恒

- input document 数量等于 output documents 覆盖数量；不得静默漏文件。
- 每个 TextUnit 的 document 引用可解析。
- 每个 entity/relationship/covariate 的 TextUnit 引用可解析。
- 每个 relationship 的 source/target 都能解析到同一 run 的 entity title。
- 不允许跨 project、跨 run 或孤儿引用。
- 检查重复 TextUnit ID 是否跨不同 document/source。
- 报告同名不同类型 raw entity collision，不得用旧 resolver 静默修复。

### 10.4 Communities 和 reports

必须满足：

```text
set(community_reports.community) == set(communities.community)
```

同时校验：

- community ID、level、parent、children 按 3.1.1 层级契约闭合；根节点允许官方 sentinel（通常为 `-1`），不得把根 sentinel 当悬空引用。
- entity_ids、relationship_ids、text_unit_ids 可解析。
- `use_lcc=false`，所有有边实体按当前配置得到合理覆盖。
- 每个非空 community 恰有当前 run 的 report。
- `findings` 为合法列表/结构值，元素满足 3.1.1 schema。
- `full_content` 为非空的人读文本，不按 JSON 解析。
- `full_content_json` 可 JSON parse；JSON 的 title/summary/findings 与同一 report 行一致，JSON `rating` 等于 Parquet `rank`，JSON `rating_explanation` 等于 Parquet 同名列。
- 不允许 report 跨 run、跨 project 或引用不存在的实体/关系。

GraphRAG 可能在报告 JSON 出错时记录错误并丢弃 report，但 CLI 最终仍退出 0；因此上述一一对应是 ready 硬门禁。

### 10.5 LanceDB

至少校验 GraphRAG 3.1.1 实际生成的三类向量表：

```text
text_unit_text
entity_description
community_full_content
```

以上是 logical embedding target。物理 LanceDB 表名由 `vector_store.index_schema.<logical>.index_name` 决定；validator/query worker 必须从 resolved settings 读取。若 settings builder 将三个 `index_name` 显式固定为上述同名值，必须用配置契约测试证明。

校验：

- row count 与对应 Parquet 目标一致。
- ID 集合一致。
- 每个向量维度等于 settings 的实际 embedding dimension。
- 不含 NaN/Inf。
- 模型 digest 和向量 manifest 一致。
- query adapter 只打开当前 run 的 LanceDB。

### 10.6 workflow ledger 和日志

十个 Standard workflow 都必须有显式终态和产物检查。

以下日志模式必须使 validator 失败：

```text
error extracting graph
No report found for community
Error processing community
```

还必须识别大小写/结构化日志等价错误，而不是只做三个区分大小写的字符串搜索。

进程退出 0 但存在上述错误、缺表、少 report 或少向量时，不得 ready。

### 10.7 corruption tests

自动复制一个最小合法 fake bundle，并分别破坏：

- 删除 manifest 文件。
- 篡改 SHA。
- 删除 Parquet 一行。
- 重复 ID。
- dangling document/TextUnit/endpoint。
- relationship 指向不存在实体。
- community parent/child 悬空。
- 删除一个 community report。
- 伪造 report community。
- 两个 communities 具有相同 report full_content 并导致 ID 碰撞。
- 删除一个 LanceDB 向量。
- 改变向量维度。
- 伪造 workflow success。
- 修改 source revision/hash。
- 用旧 run 冒充 fresh。
- 路径穿越。
- symlink 逃逸。
- 只修改 summary 而不修改原始产物。

每一种损坏都必须让独立 validator 非零退出。

---

## 11. 连续执行协议

### 11.1 状态账本

在仓库允许的本地工作目录下维护任务状态，例如：

```text
.codex/work/graphrag-native/state.yaml
.codex/work/graphrag-native/evidence/
```

如果 `.codex/` 由仓库规则禁止写入，则使用同仓库内明确忽略的替代目录。先检查 `AGENTS.md` 和 `.gitignore`。

状态至少记录：

```yaml
base_sha: null
remote_head_at_start: null
initial_worktree_status_hash: null
current_phase: P0
phase_status: pending
last_successful_gate: null
commands: []
test_counts: {}
blocked_gates: []
candidate_identity: {}
native_run_ids: []
```

### 11.2 每阶段统一循环

每阶段严格执行：

```text
读取阶段要求
→ 检查用户工作树和前序证据
→ 写失败测试/characterization test
→ 最小范围实现
→ targeted tests
→ 相关 integration/fault tests
→ 前序回归
→ diff/call-path/secret/path 自审
→ 写阶段证据与真实 test counts
→ 门禁通过则进入下一阶段
```

禁止：

- 先写全部实现，最后一次性补测试。
- 通过 `xfail`、`skip`、删除断言或放宽门禁使测试“变绿”。
- 因某个 live gate 缺权限而停止所有不依赖 live 的工程阶段。
- 只记录命令，不记录退出码、collected/passed/failed/skipped。
- 把 0 collected 或关键 marker 全 skip 当通过。

### 11.3 候选代码身份

未提交代码生成的 run 不得称为某个 commit 的结果。至少保存：

```text
BASE_SHA
WORKTREE_DIFF_HASH
CODE_TREE_HASH
runtime_env_hash
identity_cutoff_at
candidate_sha: null
```

`WORKTREE_DIFF_HASH` 必须覆盖：

- 相对 BASE_SHA 的 tracked patch；
- 任务相关 untracked 文件；
- path、mode、symlink target、size、SHA-256。

`CODE_TREE_HASH` 必须覆盖实际参与运行的源码、Prompt、settings generator、migration、脚本、依赖锁与配置。

生成 runtime artifact 后产生的 logs/output/cache 可列入显式 generated exclusions，但不能借此排除会影响运行的代码。

---

## 12. P0：冻结基线、版本和调用边界

### 目标

确定执行时真实仓库状态，避免在错误分支、错误 GraphRAG 版本或用户脏工作树上盲改。

### 必做

1. 读取所有适用 `AGENTS.md`。
2. 记录 remote HEAD、local HEAD、branch、worktree。
3. 不覆盖用户已有修改。
4. 若 remote HEAD 不同于审计锚点，先审查新增 diff。
5. 建立 Stable/Enhanced 调用路径与不变量快照。
6. 验证 Python、GraphRAG CLI/API 实际版本。
7. 识别实际下一个 migration 版本。
8. 建立阶段状态账本。

### 命令

```bash
git fetch origin fix/shadow-relation-quality-v2
git status --short --branch
git rev-parse HEAD
git rev-parse origin/fix/shadow-relation-quality-v2
git log -1 --oneline origin/fix/shadow-relation-quality-v2

test "$(git branch --show-current)" = "fix/shadow-relation-quality-v2"
test "$(git rev-parse HEAD)" = \
  "$(git rev-parse origin/fix/shadow-relation-quality-v2)"

git diff --stat
git diff --name-status

rg --files -g "AGENTS.md" . ..

rg -n \
  "STABLE_STRATEGY|ENHANCED_SHADOW_STRATEGY|normalize_strategy|submit_ingestion|submit_project_ingestion|recover_interrupted_jobs" \
  knowledge_forge tests

rg -n \
  "graphrag_community|community_reports|global_search|HugeGraph|VectorStore|canonical_entities|relation_assertions" \
  knowledge_forge tests

"$KF_PYTHON" -m pip check
"$KF_PYTHON" -c \
  "from importlib.metadata import version; print(version('graphrag')); assert version('graphrag') == '3.1.1'"
"$KF_PYTHON" -m graphrag --help
"$KF_PYTHON" -m graphrag index --help
"$KF_PYTHON" -m graphrag query --help
```

`KF_PYTHON` 应解析为仓库实际使用的 Python，例如：

```bash
export KF_PYTHON="$PWD/.venv/bin/python"
export GRAPHRAG_NATIVE_PYTHON="$KF_PYTHON"
```

如果没有 `.venv`，按仓库 README/CI 建立兼容环境，不要擅自升级依赖。

### 基线测试

先定位实际测试名，再运行与多策略、GraphRAG BYOG、reports、global search、Hubble、revision、migration 相关的现有测试。至少保留：

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model and not hugegraph_integration" \
  tests/test_multi_strategy_construction.py \
  tests/test_graphrag_community.py \
  tests/test_graphrag_reports_global_search.py \
  tests/test_hubble_query.py
```

若文件名在新 HEAD 已改变，使用 `rg --files tests | sort` 定位等价测试，记录替换原因。

### 门禁

- repo/branch/remote HEAD 与授权目标一致。
- 当前 branch 断言和 `HEAD == origin/fix/shadow-relation-quality-v2` 断言均成功；若不一致，停止并让用户决定是否切换、更新或保留本地提交。
- 用户已有修改已记录且未覆盖。
- 用户 dirty diff 若与本任务必须修改的行/符号重叠，先暂停解决冲突；不重叠的修改保留并纳入初始工作树记录。
- GraphRAG 实际版本为 3.1.1。
- CLI/API 入口已实测，不凭记忆。
- Stable/Enhanced 基线测试真实通过。
- 当前正式表、策略指针、默认查询 fixture 和关键图状态已冻结。
- 阶段证据保存完整。

未通过不得进入 P1。

---

## 13. P1：第三策略契约与显式 executor registry

### 目标

先消除“非 Enhanced 就按 Stable 运行”的隐式派发，再允许 `graphrag_native` 成为合法策略。

### 实现

1. 在 `construction_strategies.py` 增加：

```text
GRAPHRAG_NATIVE_STRATEGY = "graphrag_native"
```

2. 将服务派发改成显式 registry，例如：

```text
stable → StableExecutor
enhanced_shadow → EnhancedShadowExecutor
graphrag_native → GraphRAGNativeExecutor
```

3. 未知策略、未注册 executor、策略与 scope 不匹配必须 fail closed。
4. `graphrag_native` 只支持 project-wide full snapshot build。
5. 只拒绝“对单个文档执行 native 图构建”的动作并返回 4xx，不回退 Stable。普通文档上传、解析、OCR、structure/revision 创建仍按共享源文档流程成功；它们只使现有 native run stale，并要求下一次项目级全量构建。
6. 首阶段可以用 native stub 抛出 `NotImplemented`/明确 unavailable，但不得暴露成可成功运行。
7. `recover_interrupted_jobs` 同样使用 registry，不允许二元 `else`。
8. 默认策略和旧 API 输出保持不变。

### 测试

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model" \
  tests/test_multi_strategy_construction.py \
  tests/test_graphrag_native_strategy.py
```

新增 characterization：

- unknown strategy 失败。
- native stub 绝不调用 Stable/Enhanced。
- Stable 原 fixture 完全不变。
- Enhanced 原 fixture 完全不变。
- native 单文档图构建请求被拒绝；普通上传/解析/revision 创建成功并使 native snapshot stale。
- recovery 不把 native job 当 Stable。

### 门禁

```text
unknown → explicit error
native → native stub only
stable → original stable executor
enhanced → original enhanced executor
implicit else stable branches = 0
```

通过后自动进入 P2。

---

## 14. P2：Additive migration、run 状态机、lease 与 CAS

### 目标

建立独立 native 控制面，不污染旧实体/关系/图表。

### 实现

1. 选择执行时实际下一个 migration 编号。
2. 不修改任何已应用 migration 内容/checksum。
3. 新增第 7 节控制表、FK、索引、CHECK 和 CAS CRUD。
4. 只升级语义上确实保存“所有可选构建策略”的旧 CHECK 约束；先审查每张表的职责，不能因为列名叫 `strategy_key` 就机械加入 native。
5. 文档级 build 表若只表达 Stable/Enhanced 文档构建，不加入 native；native 文档快照由 `graphrag_native_run_documents` 表达。
6. `project_strategy_graphs` 仍只表示 HugeGraph mirror；不得成为 native ready 真相。首期不创建 native HugeGraph row，也不为它扩展 CHECK；若未来另加可选镜像，只能标识独立 projection 状态。
7. migration 如需扫描大表，不得标为 startup-safe。
8. 加入 lease acquire/renew/release、generation fencing、stale/superseded。
9. 路径列必须存相对安全路径。
10. active pointer 更新必须同时校验 validator hash、source freshness 和 generation。

### 测试矩阵

```bash
"$KF_PYTHON" scripts/manage_migrations.py plan

"$KF_PYTHON" scripts/run_disposable_postgres_tests.py \
  --strict-markers -m integration -rA \
  tests/test_graphrag_native_migrations.py \
  tests/test_graphrag_native_postgres.py \
  tests/test_multi_strategy_construction.py

"$KF_PYTHON" scripts/manage_migrations.py status
```

必须覆盖：

1. 空库从零安装。
2. 带 Stable/Enhanced 代表数据的 BASE schema 升级。
3. 重复 apply 幂等。
4. 中途异常事务回滚。
5. migration lock/并发执行。
6. checksum tamper 失败关闭。
7. 旧表 row count 和关键内容 hash 不变。
8. 跨 project/document/revision FK 被拒绝。
9. 同项目 generation fencing。
10. 晚到 worker CAS 失败。
11. 新 run 失败不替换旧 active。

### 门禁

- 所有 migration 测试通过且至少有真实 integration test collected/passed。
- 旧 migration checksum 未变。
- Stable/Enhanced 旧数据和读写路径不变。
- native state 不能由 process exit 直接写成 READY。
- 无 native 数据写进旧实体/关系/vector 表。

通过后自动进入 P3。

---

## 15. P3：结构感知输入、source map、配置与 artifact 骨架

### 目标

在不调用模型的情况下，把当前项目 revision 确定性转换成官方 GraphRAG input，并建立可独立验证的 bundle 骨架。

### 实现

1. 实现 `GraphRAGNativeInputBuilder`。
2. 只读取 active revision 和持久化 structure blocks。
3. 项目所有非删除文档按稳定顺序冻结。
4. 生成官方支持的 Parquet input。
5. 生成 input manifest 和 source-map sidecar。
6. 实现表格行、列表项、标题层级、OCR/page 的确定性转换。
7. 给重复文本加入稳定 source identity，并防止 source ID 被抽成实体。
8. 无结构块、source hash 不一致、revision 已变时 fail closed。
9. 实现 `GraphRAGNativeSettingsBuilder`，从实际 3.1.1 init/config model 生成配置。
10. 实现 artifact manifest 基础、路径安全和 redaction。
11. 不接 live 模型。

### 测试

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model" \
  tests/test_graphrag_native_inputs.py \
  tests/test_graphrag_native_config.py \
  tests/test_graphrag_native_artifacts.py \
  tests/test_table_structure_extraction.py

"$KF_PYTHON" scripts/validate_graphrag_native_bundle.py --help
```

fixture 至少包含：

- 中文普通段落。
- 标题与多段正文。
- 多行、多级表头表格。
- 国家/矿物/产量/储量/单位/年份。
- 列表中的相邻国家。
- 同名不同类型实体。
- 两个文档中的完全相同正文。
- 重复页眉页脚。
- source metadata 注入攻击。
- 缺 structure blocks。

### 门禁

- 相同输入两次生成 byte-for-byte 一致。
- 所有 active revisions 恰好一次进入 input manifest。
- 表格不跨行，列表不跨 item。
- source map 对每个 input document 全覆盖。
- GraphRAG input schema 可由 3.1.1 `ParquetFileReader`/实际 input reader fixture 解析；`index --dry-run` 不执行 input workflow，不能证明这一点。
- 使用 3.1.1 Parquet reader 的只读 fixture test 后，reader 返回的 TextDocument ID/title 集合与 input manifest 完全一致；P3 不要求尚未执行的 `output/documents.parquet`。
- settings 无明文 secret，无自定义 workflows。
- `use_lcc=false`，固定 seed，并发初始为 1。
- 路径穿越、symlink、缺文件、hash/schema/count 错误均失败。
- old chunks/canonical/BYOG 调用为 0。

通过后自动进入 P4。

---

## 16. P4：GraphRAG Standard 子进程 runner 与 provider doctor

### 目标

建立可靠的 GraphRAG 3.1.1 执行边界、失败协议、取消和审计，不把 CLI 退出码当业务成功。

### 实现

1. `GraphRAGNativeRunner` 使用 argv 数组和受控环境启动子进程，不拼 shell 字符串。
2. cwd 固定为当前 staging run root。
3. 记录 stdout/stderr 结构化日志、PID、attempt、时间和退出码。
4. timeout/cancel 时终止整个 process group，并确认无残留子进程。
5. init 只允许空 staging root。
6. 先运行带 `--skip-validation` 的 config parse dry-run，再运行不带该参数的 completion/embedding 连通性 dry-run。
7. doctor 校验非空 completion、有限数值向量和稳定维度，并把 dry-run 只在内存同步的 embedding dimension 写回最终 settings。
8. live/fake provider 状态分离。
9. retry 只重试明确 transient fault；配置、协议、schema、missing output 不无限重试。
10. run-local cache，禁止跨 run。
11. secret 只从环境传入，日志与 exception redaction。
12. 保存 GraphRAG resolved package path/version 和主项目 module path。
13. GraphRAG CLI 不自动保证逐模型调用 ledger。若需要 `model-call-ledger.jsonl`，使用不修改上游 parser 的受控 instrumentation，例如 run-local provider audit proxy 或经过测试的 LiteLLM metrics callback；只记录 call ID、阶段、模型、时间、状态、重试、token/usage（可得时）和 request/response hash，不记录 secret，也不默认导出 prompt/response 原文。
14. 如果只能取得官方聚合 metrics/cache manifest，文件必须命名并声明为 aggregate，不能伪造成逐调用 ledger；缺少可靠 extraction/report/query 非零调用证据时，相关 live 状态和 `production_eligible` 保持 false。

### 测试

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model" \
  tests/test_graphrag_native_config.py \
  tests/test_graphrag_native_runner.py \
  tests/test_graphrag_native_artifacts.py

"$KF_PYTHON" scripts/graphrag_native_doctor.py \
  --python "$GRAPHRAG_NATIVE_PYTHON" \
  --config-only
```

fake subprocess 必须覆盖：

- 0 exit + 完整产物。
- 非零 exit。
- 0 exit + 缺产物。
- 0 exit + error log。
- timeout。
- cancel。
- truncated stdout。
- malformed structured event。
- transient retry 后成功。
- terminal error 不重试。
- secret 出现在 provider exception 时被清理。

如已获得模型权限，额外运行到新临时目录：

```bash
export GRAPHRAG_SMOKE_ROOT
GRAPHRAG_SMOKE_ROOT="$(mktemp -d /tmp/kf-graphrag-native-smoke.XXXXXXXX)"

"$GRAPHRAG_NATIVE_PYTHON" -m graphrag init \
  --root "$GRAPHRAG_SMOKE_ROOT" \
  --model "$CHAT_MODEL" \
  --embedding "$EMBED_MODEL"

"$GRAPHRAG_NATIVE_PYTHON" -m graphrag index \
  --root "$GRAPHRAG_SMOKE_ROOT" \
  --method standard \
  --dry-run \
  --verbose
```

不要把手工 smoke 目录放进 repo。

### 门禁

- runner 只调用 3.1.1 CLI。
- 主 Web worker 不被 GraphRAG cwd/global state 污染。
- embedding 实际维度已探测并持久化。
- 所有失败协议正确映射到非 READY 状态。
- 取消无残留子进程。
- secret 扫描为 0。
- fake 工程门禁通过。
- 无 live 权限时 `model_connectivity_live_verified=false`，但可进入 P5 的 fake/fixture 实现。

---

## 17. P5：原生图抽取与完整 Standard workflow

### 目标

让 GraphRAG-native 从新 input 运行完整 Standard index，产出原生实体和语义关系，不接入任何旧抽取链。

### 实现

1. `GraphRAGNativeExecutor` 冻结 source snapshot。
2. 创建新的 staging attempt。
3. 生成 settings、Prompt profile、input、source map。
4. 执行十步 Standard workflow。
5. 首期 Claims disabled。
6. 原样保留 raw/final entities 和 relationships。
7. 明确 raw snapshot 已经是解析后快照，不把它当原始 model response ledger。
8. 不做旧 alias/canonical/schema/predicate 映射。
9. 对 GraphRAG 同名 title merge 行为生成 collision audit。
10. 对 source metadata 污染生成 audit。
11. 执行结束进入 `EXECUTION_COMPLETE`，等待 P6 validator。
12. input/source/config 在运行中改变则标记 `SUPERSEDED`，不得激活。
13. interrupted run 只能按明确策略重跑新 attempt 或恢复安全阶段，不覆盖旧产物。
14. 同一 source snapshot 的重建失败时，旧 active ready run 可继续查询。
15. 当前 source revision 已变化时，旧 run 立即视为 stale；即使新构建失败，native query 也返回 409，不得静默用旧内容回答新文档状态。

### 测试

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model" \
  tests/test_graphrag_native_runner.py \
  tests/test_graphrag_native_service.py \
  tests/test_graphrag_native_isolation.py \
  tests/test_graphrag_native_artifacts.py

"$KF_PYTHON" scripts/run_disposable_postgres_tests.py \
  --strict-markers -m integration -rA \
  tests/test_graphrag_native_postgres.py \
  tests/test_graphrag_native_service.py
```

### 门禁

- fake/fixed GraphRAG worker 可以形成完整十步 workflow ledger。
- entities/relationships 来源明确为 fresh `extract_graph`，不是 BYOG。
- 完整 fake/live index 后，`output/documents.parquet` 的 ID/title 集合与 input manifest 完全一致，不只比较 row count。
- relationship 两端和 TextUnit references 100% 可解析。
- ID 唯一，无跨项目/跨 run。
- Claims 明确 disabled，不是假空成功。
- current `graphrag_community.py` 旧入口调用为 0。
- 所有旧实体/关系/VectorStore/HugeGraph 入口调用为 0。
- run 中断、重试、source 变化状态正确。
- Stable/Enhanced 表计数与关键 hash 不变。

通过后自动进入 P6。

---

## 18. P6：Leiden communities、community reports 与独立验证

### 目标

确保社区和报告是当前 native graph 的完整原生产物，阻止 GraphRAG 静默丢 report 后仍被标为 ready。

### 实现

1. 固定 GraphRAG seed。
2. `cluster_graph.use_lcc=false`。
3. 保存分层 community、parent/children、entity/relationship/TextUnit membership；根 parent 按 3.1.1 sentinel 处理。
4. 保存每个 community report 的结构化 JSON 和 full content。
5. validator 从原始表重算引用闭包。
6. reports 与 communities 必须一一对应。
7. report model call 与 extraction model call 分别统计。
8. 零输入在 GraphRAG 前结束为非 READY `EMPTY_INPUT`；非空输入但零关系按 3.1.1 原生限制进入 `NO_GRAPH_UNSUPPORTED`/`FAILED`，旧 active run 保持可读。
9. 缺 report、坏 JSON、dangling membership 或错误日志导致失败。
10. 规范化成员集合用于重复运行稳定性比较，不比较随机 ID。

### 测试

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model" \
  tests/test_graphrag_native_artifacts.py \
  tests/test_graphrag_native_runner.py \
  tests/test_graphrag_native_service.py

"$KF_PYTHON" scripts/validate_graphrag_native_bundle.py \
  --root tests/fixtures/graphrag_native/valid_bundle
```

### 门禁

- `set(reports.community) == set(communities.community)`。
- community 层级和所有 references 100% 合法。
- 非最大连通分量没有被默认丢弃。
- 零输入和零关系图均为明确的非 READY、不可查询终态；不得伪造原生空 bundle。
- 任何 community/report partial failure 都不能 READY。
- corruption suite 全部非零退出。
- fake report 只证明工程；无 live 时：

```text
community_detection_verified = fake_verified
community_reports_live_verified = false
```

通过后自动进入 P7。

---

## 19. P7：LanceDB 与 Local/Global/DRIFT/Basic 查询

### 目标

直接消费当前 native run 的官方 Parquet/LanceDB，提供四种 GraphRAG 查询模式，不经过 pgvector/HugeGraph。

### 实现

1. 完成 `generate_text_embeddings`。
2. 校验三类向量表与官方输出一致。
3. 实现隔离 query worker；启动时验证 GraphRAG 版本和 run manifest hash。
4. 支持：

```text
local
global
drift
basic
```

5. 使用 GraphRAG 3.1.1 版本锁定的 Python API，不复制旧 DataFrame adapter；启动时断言实际函数签名和 `(answer, context_data)` 返回契约。
6. worker 输出受版本控制的 JSON：

```text
answer
mode
run_id
context tables
context text
context records: table + context_id + resolved native_id + source-map reference
parsed answer references（若答案确实包含可解析引用）
model usage（仅来自明确 instrumentation；否则为 unavailable）
latency
warnings
```

7. 3.1.1 context builder 中的 `id` 通常是 short/human-readable ID，不是 Parquet native `id`。serializer 必须按 table 类型映射并同时保留 `context_id` 与 `native_id`：
   - entity/relationship context `id` → 对应 final table `human_readable_id` → native `id`；
   - community/report context `id` → `community` short ID → community/report native `id`；
   - TextUnit/Basic Sources context `id` → query worker 按 3.1.1 `read_text_units()` 构造 DataFrame 后的 `reset_index()` position → TextUnit native `id`。3.1.1 这里不会直接读取 `text_units.human_readable_id`；worker 必须冻结同一排序/加载过程，并额外验证该 position 与 final table `human_readable_id` 的关系。两者不一致或无法唯一证明时 fail closed，不能猜测映射。
8. 每次映射必须恰好一条；0 或多条匹配 fail closed。解析后的 native ID 才能回到同一 run 的 entity/TextUnit/community/source map；答案内引用另行解析，并验证它是否指向已选 context 及是否支持对应主张。
9. stale、partial、failed、跨项目 bundle fail closed。
10. Global 只读当前 run 的 entities、communities、community reports，且不打开向量库。
11. Local/DRIFT/Basic 按第 4.4 节依赖只打开当前 run 的 Parquet/LanceDB。
12. map/reduce 任一步失败不能返回 success。
13. HugeGraph 不可达不能影响 native query。
14. latency 由 query worker 外部计时。公开 Python API 不返回 token/model usage；如需该字段，必须从受控 LiteLLM metrics writer、run-local audit proxy 或其他已测试 instrumentation 取得，缺失时标为 `unavailable`，不得伪造。
15. 每个 DRIFT query 必须新建 `DRIFTSearch`/`QueryState`，或使用一 query 一次性子进程；禁止跨用户请求复用 engine。3.1.1 会把 primer/action graph 保留在实例状态中，复用会造成第二个问题继承第一个问题的上下文。
16. Global/DRIFT Search 3.1.1 可能捕获内部异常并仍返回结果；worker 必须捕获结构化日志/回调状态并把下列错误映射为失败，同时区分官方真正的 NO_DATA 回答：

```text
Exception in _map_response_single_batch
Error parsing search response json
Exception in reduce_response
Exception in _asearch
No answer found for query
No follow-up actions found
```

`No answer found for query` 始终表示该 DRIFT action 没有可用回答，必须把本次 query run 标为协议失败，即使没有额外 exception。`No follow-up actions found` 只有在达到合法 termination/depth 边界并满足 primer/action contract 时才可接受，其他情况失败。若 adapter 改用 lower-level `.search()` 而不是版本锁定的 streaming wrapper，必须增加同等的 `_asearch` 静默异常检测。

### 测试

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model" \
  tests/test_graphrag_native_query.py \
  tests/test_graphrag_native_artifacts.py \
  tests/test_graphrag_native_isolation.py
```

故障注入：

- LanceDB 少一行。
- 向量维度错误。
- query run 与 bundle run_id 不同。
- context short ID 在对应 final table 中 0 匹配。
- context short ID 在对应 final table 中多匹配。
- Basic Sources short ID 被错误当成 native TextUnit ID。
- TextUnit Parquet 重排后 reset-index position 与 human_readable_id 不一致。
- source 已 stale。
- Local 缺 entity vector。
- DRIFT 缺 community vector。
- Basic 缺 text-unit vector。
- Global 缺一个 report。
- Global map exception 被 3.1.1 转为空 score。
- Global response JSON parse warning 后跳过。
- Global reduce exception 返回空字符串。
- 合法 NO_DATA 与上述内部失败的区分。
- DRIFT follow-up LocalSearch `_asearch` 异常后返回空 SearchResult。
- DRIFT 合法无 follow-up 与异常导致无 follow-up 的区分。
- 同一 worker 连续执行两个内容不同的 DRIFT query；第二次开始前 `QueryState.graph` 必须为空、primer 必须重新执行，answer/context 不得包含第一问的状态或来源。
- worker timeout/崩溃/坏 JSON。
- 单 GPU profile 下 fake/live provider 观测最大并发必须为 1；证明 `primer_folds`、`drift_k_followups` 和 worker semaphore 生效。
- HugeGraph connection refused。
- pgvector monkeypatch 调用即失败。

### 门禁

- 向量 rows/IDs/dimension/model digest 100% 一致。
- 四种模式 fixture contract 通过。
- context record IDs 100% 同 run；可解析的答案引用 100% 指向同 run 的已选 context。
- HugeGraph/pgvector 调用为 0。
- 查询失败码和 retry 语义明确。
- live 查询未运行时不能设置 `native_queries_live_verified=true`。

通过后自动进入 P8。

---

## 20. P8：Service、API、UI 与三策略黑盒路由

### 目标

让用户真正可以选择第三种方法、查看 native 状态/语义图/社区/报告并查询，同时不改变旧策略默认行为。

### 实现

1. 策略选择增加：

```text
GraphRAG Native（原生语义图与社区）
```

2. native 只支持 project-wide build。
3. 文档新增/删除/revision 变化将 active native run 标记 stale，但不就地更新。
4. 增加明确 API，按现有路由风格调整：

```text
GET  /api/projects/<id>/graphrag-native/status
POST /api/projects/<id>/graphrag-native/build
POST /api/projects/<id>/graphrag-native/cancel
GET  /api/projects/<id>/graphrag-native/graph
GET  /api/projects/<id>/graphrag-native/communities
GET  /api/projects/<id>/graphrag-native/community-reports
POST /api/projects/<id>/graphrag-native/query
```

5. `/graph/view?...strategy=graphrag_native` 打开应用内 native viewer，不进入 Hubble。
6. native graph viewer 从当前 bundle 读取限量 nodes/edges；必须分页/限量，防止一次加载全图。
7. UI 把关系展示为：

```text
SOURCE — DESCRIPTION — TARGET
weight / TextUnit provenance
```

8. 显示 run_id、source freshness、GraphRAG version、models、community/report/vector 状态。
9. `READY` 与 HugeGraph projection 状态完全分离。
10. 增加可加法演进的 capability/degraded 健康信息或 native 专属 preflight：
    - 不破坏现有整体 health 的 HTTP/字段契约。
    - Stable/Enhanced 依赖 HugeGraph 的能力单独显示。
    - native executor/query 不得调用“要求 HugeGraph 正常”的全局 preflight。
    - native build/query 不因 HugeGraph down 而被自身依赖检查阻止。
11. 旧 `/api/search` 不接受 native 后又偷偷执行 pgvector；使用独立 native query endpoint。
12. 没有 active native run 返回明确 404/409；stale 返回 409；模型/LanceDB 暂不可用返回 503。

### 测试

```bash
"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model and not hugegraph_integration" \
  tests/test_graphrag_native_service.py \
  tests/test_graphrag_native_web.py \
  tests/test_graphrag_native_query.py \
  tests/test_graphrag_native_isolation.py \
  tests/test_multi_strategy_construction.py \
  tests/test_hubble_query.py
```

### 门禁

- 三策略黑盒路由全部明确。
- 默认策略与旧 UI/API fixture 不变。
- unknown strategy 明确错误。
- native 失败不回退。
- native view/query 不调用 Hubble、pgvector 或旧 global search。
- 跨项目 run/bundle/query 被拒绝。
- HugeGraph down 时 native status/view/query 能按自身能力工作。
- native 文案不声称固定谓词或精确 span。

通过后自动进入 P9。

---

## 21. P9：不可变证据包、独立导出与损坏验证

### 目标

让任何人不依赖运行时数据库 summary，就能验证本次构建到底使用了什么代码、输入、模型、配置和原生结果。

### bundle 至少包含

```text
run-manifest.json
candidate-identity.json
candidate-source-manifest.json
candidate-worktree.patch
candidate-source-archive.tar.zst
runtime-environment.json
input-documents.parquet
input-documents-manifest.jsonl
source-map.parquet
settings.redacted.yaml
prompt-manifest.json
workflow-ledger.jsonl
model-call-ledger.jsonl（仅在逐调用 instrumentation 已验证时）
model-metrics.jsonl（否则保存可取得的聚合指标并明确 capability）
logs/
output/documents.parquet
output/text_units.parquet
output/entities.parquet
output/relationships.parquet
output/raw_entities.parquet
output/raw_relationships.parquet
output/communities.parquet
output/community_reports.parquet
output/covariates.parquet（若存在）
output/graph.graphml（若存在）
output/lancedb/
lancedb-manifest.json
query-smoke.jsonl
migration-report.json
native-isolation.json
stable-enhanced-invariance.json
metric-summary.json
validation-report.json
audit-report.md
```

原始 cache 可能包含敏感内容。默认保存在受保护的 run-local 目录；导出包中保存必要 hash/manifest，除非现有安全策略明确允许导出原文 cache。

### 实现

1. `export_graphrag_native_bundle.py` 只导出某个不可变 run。
2. 重新计算所有文件 hash/schema/count。
3. 禁止用 PostgreSQL/HugeGraph 投影回填缺失原生文件。
4. 导出后再次运行独立 validator。
5. validator 不信任 `run-summary.json` 的结论。
6. 保存所有实际命令、退出码、test counts 和 skipped markers。
7. 扫描明文 API key、Authorization header、密码和私有 endpoint credential。
8. 运行第 10.7 节 corruption suite。

### 命令

```bash
"$KF_PYTHON" scripts/export_graphrag_native_bundle.py \
  --project-id "<PROJECT_ID>" \
  --run-id "<RUN_ID>" \
  --output "<NEW_EMPTY_OUTPUT_DIR>"

"$KF_PYTHON" scripts/validate_graphrag_native_bundle.py \
  --root "<NEW_EMPTY_OUTPUT_DIR>"

"$KF_PYTHON" scripts/check_graphrag_native_freshness.py \
  --project-id "<PROJECT_ID>" \
  --run-id "<RUN_ID>"

"$KF_PYTHON" -m pytest --strict-markers -q \
  -m "not integration and not live_model" \
  tests/test_graphrag_native_artifacts.py
```

### 门禁

- 合法 bundle 独立验证通过。
- online freshness gate 在当前 PostgreSQL snapshot 上通过；改变任一 active revision 后必须失败。
- 每个 corruption fixture 失败关闭。
- secrets 扫描为 0。
- manifest 可离线重算。
- old artifact 不能冒充 fresh。
- `native-isolation.json` 证明禁用入口调用为 0。
- `stable-enhanced-invariance.json` 可从基线重算。

通过后自动进入 P10。

---

## 22. P10：fresh live E2E、固定 Gold、A/B 和三次重复

### 目标

证明真实模型下 GraphRAG-native 从源 revision 到社区和查询完整运行，并用语义关系/社区/查询 Gold 判断质量，而不是按实体或关系数量自证。

### 22.1 固定 smoke corpus

先用小型固定 corpus 验证协议，至少覆盖：

- 中文段落实体与清晰关系。
- 表格同一行的国家、矿物、产量、储量、单位和年份。
- 不同表格行不可串联。
- 相邻列表项不可互连。
- 同名不同类型实体。
- 重复 header/footer。
- 明确否定或不确定关系。
- 完全相同正文出现在不同文档。

smoke 必须真实触发：

```text
graph extraction model calls > 0
community report model calls > 0
embedding calls > 0
query model calls > 0
```

### 22.2 fresh 项目 E2E

必须：

1. 从项目源文档/active revisions 开始。
2. 新建空 run/output/cache。
3. 记录实际 import 的 Knowledge-Forge 和 GraphRAG 模块路径。
4. 完成十步 Standard workflow。
5. 完成独立 validator。
6. 完成 Local/Global/DRIFT/Basic query smoke。
7. HugeGraph 关闭或不可达时仍成功。
8. 前后比较 Stable/Enhanced 表、图、默认查询和 activation。
9. 不使用历史 Parquet、BYOG、cache 或 artifact。

建议提供 runner：

```bash
RUN_LIVE_MODEL_TESTS=1 \
"$KF_PYTHON" scripts/run_graphrag_native_pilot.py \
  --project-id "<PILOT_PROJECT_ID>" \
  --fresh \
  --validate \
  --query-smoke
```

不得在代码中硬编码真实 project ID、模型 endpoint 或密钥。

### 22.3 质量评估口径

GraphRAG-native 关系评估：

```text
(source, target, description, provenance)
```

不得直接套用 Enhanced 的 predicate micro-F1 和 direction accuracy。

至少分别报告：

- TextUnit 级 entity presence precision、recall、F1，匹配键为 `(document_id/text_unit_id, normalized_entity_title, entity_type)` 或经人工裁定的语义等价键；
- relationship endpoint-pair precision、recall、F1；
- relationship description support precision；
- cross-row/cross-list contamination；
- source metadata contamination；
-同名实体 collision；
- community membership coherence；
- community report supported-claim precision；
- critical hallucination count；
- query selected-context ID integrity、answer-reference parse rate 和 answer-claim support；
- unanswerable query abstention；
- tokens/calls/retries/latency/cost。

GraphRAG 原生输出没有精确 mention span，因此不得把上述 TextUnit/source-map presence 称为字符级 mention；exact-span 指标标记 `not_applicable`。Claims disabled 时不报告 claim quality。

### 22.4 预注册硬门槛

如果仓库已有人工 Gold，以已冻结 Gold 为准，不得看完结果后改答案。没有 Gold 时只能生成待人工审核包，不能生产放行。

生产资格使用的 Gold 至少满足：

- dev 与 holdout 按 document 完全隔离；同一文档的相邻 TextUnit、表格行或重叠内容不得跨集合。
- Prompt tune、few-shot、阈值和错误分析只能使用 dev，打开 holdout 前冻结代码、Prompt、模型、配置、指标和阈值。
- holdout 至少覆盖 20 个源文档、200 个已审核 TextUnits、300 个 entity-presence 判断和 200 个 semantic-edge 判断。
- community report 至少审核 20 个 communities、100 个可判定 report claims。
- query Gold 至少包含每种模式 20 个可回答问题，共至少 80 个；另有至少 20 个不可回答问题。
- 表格、列表、普通段落、重复 header、同名实体和否定/不确定性都必须有样本。
- 两名独立审核者先盲审，再由第三人或预先指定规则仲裁；保存原始判断、分歧率和一致性统计，不得只保留仲裁后结果。
- 样本不足时可以报告 pilot point estimate，但 `production_eligible=false`。

建议最低门槛：

```text
all structural/reference invariants = 100%
relationship description support precision >= 0.90
critical cross-row/list false links = 0
community report supported-claim precision >= 0.95
community report critical hallucinations = 0
query answer-claim support against selected context >= 0.95
query critical hallucinations = 0
unanswerable abstention = 100%
```

比例指标同时报告 point estimate 和 95% 置信区间：

- precision/support/abstention 使用 Wilson lower bound；
- recall、F1、community/query 聚合指标使用 document-level bootstrap，至少 2,000 次重采样；
- 生产门禁使用 holdout 置信区间下界与三次 fresh run 的最差结果，不使用最好一次或仅使用均值；
- critical hallucination/cross-row false link 的要求始终是观察数 0，并同时报告样本量。

实体/关系 recall 必须：

- 明确报告；
- 不低于冻结的共同 Gold 最低目标；
- 与 Stable/Enhanced 在同一输入和同一 Gold 上比较；
- 不能用总数量替代；
- 不能为了过线在 test set 上迭代 Prompt 后仍把该 test set 称为 holdout。

如果现有 Gold 不适合原生语义关系，先版本化新增 `graphrag_native` Gold 和 adjudication guide，并保留独立 holdout。

### 22.5 三次重复

获得真实模型和成本授权后，用相同：

```text
source snapshot
candidate identity
GraphRAG version
model digests
Prompt hashes
settings hash
seed
Gold version
```

运行三次 fresh bundle。

报告：

- 每次结果；
- mean/worst/range；
- entity/edge/community normalized stability；
- report/query support；
- calls/tokens/retries；
- p50/p95 latency；
- 失败率。

不得挑最好一次。

### live 测试命令

```bash
RUN_LIVE_MODEL_TESTS=1 \
"$KF_PYTHON" -m pytest --strict-markers -q -rA \
  -m live_model \
  tests/test_graphrag_native_live.py
```

### 门禁

- live markers collected > 0 且 passed > 0。
- fresh extraction/report/embedding/query calls 均 > 0。
- 独立 bundle validator 通过。
- Stable/Enhanced 不变量通过。
- HugeGraph down 注入通过。
- 人工 Gold 达到预注册门槛。
- 三次重复达到 worst-case 门槛。

缺任何条件时分别标记 blocked；不得伪造 `pilot_quality_passed` 或 `production_eligible`。

---

## 23. P11：恢复、回滚、运维、CI、Docker 与最终交付

### 目标

完成生产所需的恢复语义、文档和全量回归，但不自动启用、发布或删除数据。

### 实现

1. 新构建失败保留旧 active ready run。
2. lease 过期可安全恢复或新 attempt 重跑。
3. 晚到 worker 被 generation fencing 拒绝。
4. 分别实现两个 CAS 指针的回滚：
   - native active run：`N → N-1 ready run`；
   - project selected construction strategy：`graphrag_native → stable` 或 `enhanced_shadow`。
5. 两种回滚都只切指针、递增 generation 并使并发查询重新校验，不删除 bundle，也不触发旧链重建或跨策略写入。
6. source/model/config/prompt/code 改变后旧 run stale；source 已变化时旧 run 不得继续回答。
7. migration 不支持 downgrade 时提供 backup/restore/forward-fix runbook，禁止自动 DROP。
8. 提供磁盘容量、staging orphan、cache 和 retention 观测。
9. retention 工具默认 `--dry-run`，拒绝删除 active/last-ready/leased run。
10. 更新 README、AGENTS、环境示例、架构文档。
11. Docker/Compose 创建并授权：

```text
KnowledgeFiles/graphrag_native
```

不新增 PostgreSQL、HugeGraph 或独立 LanceDB 网络服务；LanceDB 是 run-local 嵌入式目录。

12. CI 增加：
    - strategy registry；
    - GraphRAG version/config contract；
    - fake Standard runner；
    - validator/corruption；
    - disposable PostgreSQL lifecycle；
    - Stable/Enhanced invariance。
13. live model tests 保持显式 marker，不在普通 CI 假运行。

### 全量命令

```bash
"$KF_PYTHON" -m py_compile knowledge_forge/*.py scripts/*.py server.py
"$KF_PYTHON" -m pip check

"$KF_PYTHON" -m pytest --strict-markers -q -rA \
  -m "not integration and not live_model and not hugegraph_integration"

"$KF_PYTHON" scripts/run_disposable_postgres_tests.py \
  --strict-markers -m integration -rA

docker compose -f docker-compose.yml config --quiet
"$KF_PYTHON" scripts/verify_docker_artifacts.py --build --health

git diff --check
git status --short
```

如果仓库实际有 lint/typecheck 命令，读取 CI/Makefile/pyproject 后补跑，不得臆造工具。

### 需要记录

- 每条命令原文。
- 退出码。
- collected/passed/failed/skipped。
- Docker image/build/health 结果。
- migration fresh/upgrade/reapply 结果。
- run ID、bundle path、manifest hash。
- blocked live/Gold 条件。
- `git diff --stat` 和任务相关文件清单。

### 门禁

- compile、pip check、unit、integration、Docker/Compose、diff check 全部通过。
- 关键 marker 不是 0 collected。
- Stable/Enhanced characterization 与 invariance 全部通过。
- docs 与实际行为一致。
- recovery/fencing/rollback/retention 测试通过。
- native active-run pointer 与 project selected-strategy pointer 的独立/并发 CAS 回滚测试通过。
- 没有自动 commit/push/PR/deploy/activate。
- `active=false`。

---

## 24. Stable 与 Enhanced 不变量

开始和结束时分别采集、比较：

```text
Stable/Enhanced 默认策略
现有 project/document strategy selections
正式 entity/relation/vector 表 row count 与代表数据 hash
Enhanced shadow 表 row count 与代表数据 hash
HugeGraph namespaces/active generations
旧 GraphRAG BYOG community runs
旧 reports/global-search 数据
默认 search API fixture
Stable ingest fixture
Enhanced ingest fixture
activation/current policy
```

共享文件允许薄适配，不要求 whole-file byte-identical；但必须：

- 进入 approved shared-adapter allowlist；
- 有调用图；
- 有 old-path characterization；
- 有前后黑盒 fixture；
- 证明 native-only 配置默认关闭；
- 证明 unknown/non-native 分支没有变化。

若 invariance 失败：

```text
停止，不得用“看起来无关”解释后继续
```

先定位并修复，再重新跑当前及全部前序门禁。

---

## 25. 必须测试的失败场景

至少覆盖：

### 输入

- active revision 缺失。
- source hash 改变。
- structure blocks 缺失。
- document 在 snapshot 后删除。
- 两文档文本完全相同。
- 表格跨行污染。
- 列表跨项污染。
- source metadata 被抽成实体。

### provider/model

- completion connection refused。
- embedding connection refused。
- wrong model name。
- empty response。
- 全部 malformed extraction tuples，最终零图并非 READY。
- 合法与 malformed tuple 混合时，记录 3.1.1 会静默跳过部分 record 的已知限制，并由 smoke/Gold 检出质量损失；不得假称 Parquet validator 能还原被跳过项。
- truncated output。
- report JSON invalid。
- timeout。
- retry exhausted。
- embedding dimension drift。

### process

- init 失败。
- dry-run 失败。
- index 非零退出。
- index 0 退出但缺表。
- index 0 退出但 error log。
- cancel。
- worker 崩溃。
- lease 过期。
- late worker。
- restart recovery。

### artifacts

- hash、schema、row count 不符。
- duplicate/dangling IDs。
- missing report/vector。
- cross-run reference。
- path traversal/symlink。
- stale source。
- forged summary。
- historical cache reuse。

### query

- no active run。
- active run stale。
- wrong project/run。
- missing LanceDB table。
- map succeeds/reduce fails。
- worker bad JSON。
- HugeGraph down。
- pgvector unavailable。

### isolation

- old extractor called。
- old resolver called。
- pair/classifier/verifier called。
- old canonical/relation/vector write。
- HugeGraph read/write。
- old BYOG export。
- fallback Stable/Enhanced。

每个失败都必须得到明确状态、错误分类和审计记录；不得转换为合法空成功。

---

## 26. 最终验收矩阵

| 层级 | 硬验收 |
|---|---|
| 策略派发 | 三策略显式 registry；未知策略失败；无隐式 Stable fallback |
| 输入 | 所有 active revision 精确冻结；结构/source map 全覆盖；无 legacy chunk fallback |
| GraphRAG fidelity | fresh Documents → TextUnits → Graph → Communities → Reports → Embeddings |
| 原生关系 | 保留 source/target/description/weight/TextUnit，不强制旧 predicate |
| 图完整性 | ID、endpoint、TextUnit provenance 100% 可解析 |
| 社区完整性 | membership、parent/children、report 一一对应 |
| 向量完整性 | LanceDB IDs/rows/dimension/model digest 100% 一致 |
| 查询隔离 | Local/Global/DRIFT/Basic 只读同一 run |
| 服务隔离 | 旧抽取/关系/pgvector/HugeGraph native 调用为 0 |
| PostgreSQL | 只作控制面；fresh/upgrade/reapply/recovery/CAS 通过 |
| HugeGraph | Stable/Enhanced 保留；native build/query 在其宕机时可运行 |
| Artifact | 独立 validator 通过；所有损坏 bundle 失败 |
| 旧策略 | Stable/Enhanced 默认 API、数据、图、查询和 activation 不变 |
| 质量 | 固定 Gold；description/report/query 支持率达标；关键幻觉为 0 |
| 成本 | workflow 级 calls/tokens/retries/latency 完整 |
| 发布 | 不自动发布、不自动激活 |

---

## 27. 完成状态必须逐项输出

最终报告必须输出下面的 YAML，并为每项给出证据或 blocker：

```yaml
implementation_complete: false
postgres_integration_verified: false
native_index_fake_verified: false
native_index_live_verified: false
community_detection_verified: false
community_reports_live_verified: false
lancedb_verified: false
native_queries_live_verified: false
artifact_validation_passed: false
stable_invariance_passed: false
enhanced_invariance_passed: false
pilot_quality_passed: false
production_eligible: false
active: false
hugegraph_projection_verified: not_applicable
```

状态含义：

- `implementation_complete`：代码、migration、文档、非 live 测试和可运行入口齐全。
- `native_index_fake_verified`：fake/fixed worker 完整验证工程链。
- `native_index_live_verified`：真实模型 fresh index 完成且 validator 通过。
- `community_reports_live_verified`：真实报告模型调用非零，完整报告覆盖通过。
- `lancedb_verified`：真实/明确标记的 bundle 向量行、ID、维度通过。
- `native_queries_live_verified`：四种真实查询完成，selected-context IDs、可解析答案引用和 claim support 校验通过。
- `pilot_quality_passed`：冻结人工 Gold 达到预注册门槛。
- `production_eligible`：不可变 candidate、live、Gold、三次重复、全量回归全部通过。
- `active`：只有用户另行授权并实际激活才可为 true；本任务结束时必须为 false。
- `hugegraph_projection_verified`：首期必须为 `not_applicable`。

缺 live 模型时，最多完成 engineering/fake/disposable integration；缺人工 Gold 时不能设置 `pilot_quality_passed` 或 `production_eligible`。

---

## 28. 暂停条件

只有下列情况可以暂停：

1. 当前仓库/分支不是授权目标，且无法安全确认目标。
2. 用户已有修改与必须修改的同一代码区域发生实质冲突，无法保留双方语义。
3. 需要 destructive database/file operation。
4. 需要非 disposable 数据库、生产模型、生产密钥或高成本重复运行的新授权。
5. 需要 commit/push/PR/deploy/activate。
6. 远端分支在 identity cutoff 后前进，继续会使目标不明确。
7. 官方 GraphRAG 3.1.1 实际 API 与本任务契约重大冲突，需要改变架构而非局部适配。
8. Stable/Enhanced invariance 无法在当前范围内恢复。
9. 发现安全问题、secret 泄漏或路径逃逸。

暂停时必须报告：

- 已完成到哪个 gate；
- 最后通过的命令和证据；
- 精确 blocker；
- 未完成项；
- 需要用户做的最小决定；
- 当前 git diff/status；
- 不得声称整个任务完成。

普通测试失败、实现 bug、依赖小差异或 fixture 缺失不是暂停理由，应在范围内修复并继续。

---

## 29. 最终交付报告模板

```markdown
# GraphRAG Native 第三策略实施报告

## 结果
- BASE_SHA:
- candidate identity:
- GraphRAG version:
- implemented phases:
- blocked phases:

## 修改
- 新增文件:
- 薄适配文件:
- migration:
- API/UI:

## 实际命令与测试
| command | exit | collected | passed | failed | skipped |

## native run
- project:
- run_id:
- source snapshot hash:
- settings/prompt/model hashes:
- artifact path:
- manifest hash:

## GraphRAG workflow
| workflow | status | rows/artifacts | model calls | errors |

## 隔离与不变量
- legacy calls:
- formal writes:
- pgvector calls:
- HugeGraph calls:
- Stable diff:
- Enhanced diff:

## 质量与成本
- entity metrics:
- relationship metrics:
- community/report metrics:
- query metrics:
- calls/tokens/latency:
- three-run stability:

## 完成状态
<第 27 节 YAML>

## Blockers
- ...

## 未授权动作
- commit: not performed
- push/PR: not performed
- deploy: not performed
- activation: not performed
```

---

## 30. 官方参考

- Microsoft GraphRAG v3.1.1：
  `https://github.com/microsoft/graphrag/tree/14a00ad88fc33cf2b52f4f113f25807556f8e25e`
- Standard indexing dataflow：
  `https://microsoft.github.io/graphrag/index/default_dataflow/`
- GraphRAG inputs：
  `https://microsoft.github.io/graphrag/index/inputs/`
- GraphRAG outputs：
  `https://microsoft.github.io/graphrag/index/outputs/`
- GraphRAG configuration：
  `https://microsoft.github.io/graphrag/config/yaml/`
- GraphRAG BYOG：
  `https://microsoft.github.io/graphrag/index/byog/`

实现时以固定的 3.1.1 实际安装包、tag source 和 CLI/API 自检为准。不得无审计升级到新版本，也不得从最新 `main` 复制不兼容配置或 API。

---

## 31. 一句话执行标准

> 新增一个真正从源文档开始、完整运行 Microsoft GraphRAG Standard pipeline 的第三策略；PostgreSQL只管运行控制，Parquet/LanceDB 是唯一算法事实源，HugeGraph与旧实体关系链不参与；每个阶段以测试和独立 artifact 校验为门禁，直到工程、真实模型、社区、查询和质量验收在当前权限范围内全部完成或被准确标记为阻塞。
