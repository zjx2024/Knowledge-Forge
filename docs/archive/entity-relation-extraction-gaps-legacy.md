# 实体、关系抽取与合并现状

本文记录当前 Knowledge Forge 实体与关系链路已经具备的能力、仍存在的质量问题，
以及这些问题对知识图谱和 GraphRAG 社区结果的影响。结论以当前代码路径和自动化测试
为准，不把历史运行库的旧统计当作当前准确率。

## 已完成能力

实体链路已经按 occurrence 保存精确半开位置、原文 surface、类型、证据和稳定
`mention_key`。候选归一采用规范名称精确匹配、PostgreSQL 索引、pgvector Top-K、
类型与名称规则、reranker 和必要时的 LLM 裁决；cannot-link、重定向和同名异义均按
fail-closed 处理。重试会优先复用已有 occurrence assignment，不因模型再次返回不同
答案而改变已确定归属。

关系链路只在当前文档 active revision 的局部证据窗口内生成候选，并对 source、chunk
和 batch 设置预算。候选必须进入 specific relation、`NO_RELATION`、
`OTHER_UNSUPPORTED`、uncertain、retryable、overflow、deferred 等互斥状态之一；覆盖
不完整会阻断 finalize、质量门和激活，未知结果不会被强制降级成 `related_to`。

正式图只消费可发布 control 投影。P1/P2 treatment 保持 shadow。GraphRAG 社区检测
读取同一正式快照，同时保留有 active mention 支撑但暂时没有关系的孤立实体。

## 当前主要问题

### 1. 实体边界仍可能偏宽

模型会把带修饰语的长短语、句子片段或复合概念整体识别成实体。这会增加低价值、仅
出现一次的规范实体，并降低跨文档复用率。精确 span 校验可以防止错位，但不能判断
语义边界是否足够简洁。

### 2. 关系召回弱于实体召回

显式、同句关系较容易抽取；跨句、跨 Chunk、隐含关系和需要领域常识的关系仍容易
遗漏。系统不会用跨文档共现补造证据，因此实体数量增长不等于关系数量同步增长，图中
可能出现较多孤立点或弱连接分量。

### 3. 关系复核偏保守

严格证据、端点、方向、类型和 verifier 检查能降低错误边，但会拒绝证据表达不够直接
或模型输出不完整的候选。日志中的“通过 + 拒绝”不一定等于候选总数；其余候选可能
进入 `NO_RELATION`、unsupported、uncertain、retryable、overflow、deferred、missing
response 或 quarantine。诊断时应查看完整漏斗，而不是只比较两个数字。

### 4. 关系类型仍可能过于泛化

当 logical schema 的谓词定义、提示示例或原文证据不足时，模型更容易集中到少量通用
关系。正式链路禁止把未知关系自动改成 `related_to`，但仍需人工 Gold 判断现有具体
谓词的覆盖面和边界是否适合目标领域。

### 5. 实体归一仍有保守拆分

保守策略能避免把同名异义实体错误合并，但缩写、全称、译名、型号变体和上下文稀疏的
同一实体仍可能分成多个 canonical。pgvector 和 reranker 已将比较限制在小候选集，LLM
只做必要的最终裁决；在缺少人工 must-link/cannot-link Gold 时，不能仅凭“合并次数”
判断质量。

### 6. 社区报告引用粒度仍较粗

Community Report 的 finding 可以解析到同一快照的 community、entity 和 relation，
但 GraphRAG 原生输出没有 Knowledge Forge 的 claim-level citation。当前适配器附加的是
社区输入级溯源，能定位原 assertion、mention、Chunk 和 active revision，却不能冒充
人工验证过的逐句事实支持。

## 对当前结果的判断

当前链路已经具备可追溯、可恢复、可测量和不污染正式图的工程基础，适合继续做 shadow
评估。实体归一已产生实际价值，关系链路也避免了以数量换准确率；但实体边界、复杂关系
召回、谓词覆盖和 report claim 支持仍需要独立人工 Gold 与真实模型验证。

因此当前状态应表述为：工程闭环可用，P1/P2 treatment 仍为 shadow；GraphRAG 社区、
Community Reports 和显式 Global Search 可以做隔离验证，但不能据此宣称已经达到生产
激活标准。
