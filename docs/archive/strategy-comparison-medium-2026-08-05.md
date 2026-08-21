# 同一文档的 Knowledge Forge 与 GraphRAG Native 对比

## 实验对象

两种策略使用完全相同的源文件，数据库中的 `doc_hash` 和文件 SHA-256 均为：

`3da98c8487580a500434f1614469595354965407a8f423b5112089362b67b151`

源文件为 `strategy-comparison-medium.txt`，1058 个解析字符、7 个章节/结构单元。两次运行均从各自项目的真实公共入口执行，没有复用另一策略的中间结果：

- Knowledge Forge：项目 56，Shadow v2 run `shadow_relation_v2_8f585a9c7a452db36f2f240fe240d6ab4cb9ca6ffd7cc4bb2eabba1502e66809`
- GraphRAG Native：项目 57，GraphRAG 3.1.1 run `grn_cda155e364d743bd8a464bad945b6af1`
- GraphRAG 构建任务：`c0ce75bb94814a5784047ea1269a1beb`，耗时 1057 秒，最终状态 `completed`

## 产物数量对照

| 项目 | Knowledge Forge | GraphRAG Native |
|---|---:|---:|
| 源文档单元 | 7 个结构化 chunk | 7 个 TextUnit |
| 实体提及/原始实体 | 113 个实体提及 | 44 个实体 |
| 规范实体/最终实体 | 59 个规范实体，6 类 | 44 个实体，主要为 ORGANIZATION/GEO/EVENT |
| 关系 | 23 条规范事实，6 个谓词 | 62 条无显式谓词类型的关系边 |
| 关系证据 | 24 条证据行，23/23 有 SUPPORTED verification | 每条边有 TextUnit 引用和生成描述，但没有独立 verifier 终态 |
| 社区 | 41 个社区，其中大量单实体社区 | 7 个社区、7 份社区报告 |
| 节点描述 | 当前 Shadow 快照不生成统一节点摘要 | 44/44 节点有 GraphRAG 描述 |

GraphRAG 的实体、关系和社区均成功写入独立 bundle：`KnowledgeFiles/graphrag_native/project-57/runs/grn_cda155e364d743bd8a464bad945b6af1`。其中 `entities.parquet=44`、`relationships.parquet=62`、`text_units.parquet=7`、`communities.parquet=7`、`community_reports.parquet=7`。

## 实体质量

### Knowledge Forge

- 保留了更细的通用类型（Organization、Location、Product、Technology、Event、GenericEntity），便于后续做谓词角色约束。
- 仍有两个已确认问题：同名“智能制造实验室”被拆成 Organization 和 Product；“合肥先进制造基地”主类型偏成 Organization，但上下文多次把它当作物理基地/地点。
- 另外保留了“供应方”“制造基地”“示范园区”等弱实体，需要继续用角色和指代门禁控制。

### GraphRAG Native

- 原始数量较少，但存在明显的中英文/翻译变体重复。例如同一对象同时出现“海岳智能制造集团”和 `HAIYUE INTELLIGENT MANUFACTURING GROUP`，“东辰研究院”和 `DONGCHEN RESEARCH INSTITUTE`，“星图软件公司”和 `XINGTUO SOFTWARE COMPANY`，“智能制造实验室”和 `SMART MANUFACTURING LABORATORY`。
- “北辰电气公司”还被拆成 `BEICHEN ELECTRICAL COMPANY`、`BEICHEN ELECTRIC COMPANY`、`BEICHEN ELECTRIC EQUIPMENT COMPANY` 等多个组织，导致关系被分散到不同节点。
- 类型粒度明显偏粗且有错位：机器视觉平台、能源管理系统、精密装配产线被标成 ORGANIZATION；装配车间和质量检测中心被标成 GEO；没有 Product/Technology 类型。
- 因此，GraphRAG 的 44 个实体不能直接解释为“比 Knowledge Forge 少 15 个无用实体”；其中一部分是语言变体重复，另一部分是类型和实体边界损失。

## 关系质量

GraphRAG 的 62 条边数量明显多于 Knowledge Forge 的 23 条事实，但它们没有统一 predicate 字段，语义主要依赖模型生成的英文 description。对 62 条边按原文逐条抽查后：

- 约 32 条有原文直接或明确隐含支持，例如总部位置、供应链运输、基地包含车间、两段设备中断因果链、委员会代表关系等。
- 约 23 条属于泛化或跨章节推断，例如把“协作网络连接研发/制造/供应/物流”展开成大量网络成员边，或把参与网络推断成“核心参与者”；这些边不一定完全错误，但已超出原文的精确端点和谓词语义。
- 至少 7 条为明确不应发布或明显不可靠的边，包括：
  - `海岳智能制造集团 → 星图软件公司`，原文明确说“不使用星图软件公司本身”；
  - `XINGTUO SOFTWARE COMPANY → HAIYUE INTELLIGENT MANUFACTURING GROUP` 与上条形成语言变体重复，且把“提供平台”提升成公司级关系；
  - `能源存储系统事业部 → 精密装配产线`，原文没有该关系；
  - `能源存储系统事业部 → 北辰电气公司`，属于同一供应链语境下的共现推断；
  - `SMART MANUFACTURING LABORATORY → ENERGY MANAGEMENT SYSTEM`，原文没有实验室监控能源管理系统的断言；
  - `SECURITY AUDIT TEAM → SHANGHAI DEMONSTRATION PARK`，原文只说团队检查能源管理系统访问日志；
  - `INDUSTRIAL CAMPUS → DEMONSTRATION PARK`，源文档没有“Industrial Campus”这一实体关系。

GraphRAG 还把“北辰电气公司供应双向变流器”“启明传感器公司是供应方”等角色语义生成成无类型边，无法区分 `supply_to`、`uses`、`produces` 和 `part_of`。这使得边数更高，但不能直接用于严格的关系审计。

Knowledge Forge 的 23 条事实全部有精确 span 和独立 SUPPORTED verification，且负向关系没有发布；不过人工核验仍发现两条方向错误：

- `located_in(合肥先进制造基地, 机器人装配车间)` 应为 `located_in(机器人装配车间, 合肥先进制造基地)`；
- `uses(力控传感器, 精密装配产线)` 应为 `uses(精密装配产线, 力控传感器)`。

所以 Knowledge Forge 当前的关系精度和可审计性更高，但方向归一和召回仍未完全收口；GraphRAG 的关系召回更宽，代价是语言变体重复、类型/谓词缺失和明显的否定句误连。

## 社区与描述

- GraphRAG 生成了 7 个有标题、摘要、发现项和引用标记的社区报告，适合做全局主题概览和问答上下文；但报告会继承并放大上面的错误边，例如把未在原文出现的网络成员、实验室与能源管理系统关系写入叙述。
- Knowledge Forge 当前有 41 个社区，很多是单实体社区，反映其 23 条保守事实形成的稀疏图；社区摘要能引用已验证事实，但主题聚类的可读性和节点描述仍弱于 GraphRAG。
- 两者的社区数量不能直接比较质量：GraphRAG 的 7 个社区来自 62 条宽松关系，Knowledge Forge 的 41 个社区来自 23 条严格事实。

## 结论

对这份同一文档，若评价“实体归一、关系方向、证据可追溯和误报控制”，Knowledge Forge 更有保留价值；若评价“自动节点描述、社区报告、全局主题发现和问答素材覆盖”，GraphRAG Native 更完整。

本次不能用“62 条边 > 23 条事实”推出 GraphRAG 质量更高。当前更准确的结论是：

1. GraphRAG 的社区和描述能力明显更强，但实体归一和关系语义过宽，且出现了可由原文否定的关系。
2. Knowledge Forge 的结构化事实链更严格、可审计，并正确拦截了多个否定关系，但仍需修复通用方向归一、跨类型实体复核和召回缺口。
3. 没有人工 Gold 标注时，不宣称 precision、recall 或综合质量提升；本报告是同文档人工核验结果。
