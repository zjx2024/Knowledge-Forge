# Knowledge Forge 文档索引

> 当前维护基线：2026-09-01。根目录 `README.md` 负责安装、运行和完整功能入口；本目录按产品状态、知识构建、智能体和运维边界拆分说明。

## 当前维护文档

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md)：当前知识构建与单知识库智能体能力、质量判断、实现边界和下一阶段；
- [`deployment.md`](deployment.md)：换机部署、Ollama/HugeGraph/reranker、远程模型 API、CPU Compose 与 systemd；
- [`knowledge-forge-current-strategy-interview-guide.md`](knowledge-forge-current-strategy-interview-guide.md)：Knowledge Forge 完整构建策略、公式、统一智能体概览与面试说明；
- [`knowledge-agent-implementation-roadmap.md`](knowledge-agent-implementation-roadmap.md)：统一知识问答智能体的现行实现、协议和从第一版到最终版的路线；
- [`naming-and-compatibility.md`](naming-and-compatibility.md)：当前代码术语和必须保留的历史存储标识；
- [`graphrag-native.md`](graphrag-native.md)：GraphRAG Native 数据边界、运行与查询；
- [`knowledge-forge-evaluation-acceptance-2026-08-11.md`](knowledge-forge-evaluation-acceptance-2026-08-11.md)：当前跨领域评估与可接受标准；
- [`migrations-and-release-safety.md`](migrations-and-release-safety.md)：迁移和发布安全；
- [`pre-live-readiness.md`](pre-live-readiness.md)：构建与智能体真实环境验证、迁移及恢复检查；
- [`project-strategy-workspaces.md`](project-strategy-workspaces.md)：项目与策略工作区隔离。

## 历史记录

`archive/` 中的文件是特定日期实验、审计或旧实施方案，只用于追溯当时判断。它们可能包含已经废弃的策略名称、质量结论和命令，不应作为当前运行手册。

## 文档维护规则

- 当前行为变化时先更新当前维护文档，不新增根目录任务提示文件；
- 一次性评测结果放入本地 `artifacts/`，不提交大体积运行输出；
- 历史审计如需保留，文件名必须带日期并移动到 `archive/`；
- 数据库物理旧名称的解释统一链接到 `naming-and-compatibility.md`。
