# Knowledge Forge 多领域测试文档

本目录包含 5 组公开文档，用于分别新建项目并测试基础 Schema 演化、实体关系抽取、跨文档实体归一和联合检索。所有文件均来自官方公开来源，不会自动导入系统。

## 建议测试顺序

1. 为每个子目录新建一个独立项目。
2. 先上传该目录中的第一个文档，记录 Schema 版本、实体类型和关系类型。
3. 再上传第二个文档，观察 Schema 是否增量演化，以及同名或近义实体是否归并。
4. 对比 PostgreSQL 中的实体归一决策与 HugeGraph 中的最终实体、关系数量。

## 01-software-protocols

建议项目名：`HTTP 协议知识图谱`

- `rfc9110-http-semantics.txt`：HTTP 语义、方法、状态码、资源和中间节点。
- `rfc9112-http11.txt`：HTTP/1.1 消息、连接、客户端、服务器和安全问题。

重点观察：Protocol、Method、StatusCode、Message、Client、Server 等实体，以及 defines、uses、sends、responds_with、obsoletes 等关系。两个 RFC 共享 HTTP、客户端、服务器等实体，适合测试跨文档归一。

来源：

- https://www.rfc-editor.org/rfc/rfc9110.txt
- https://www.rfc-editor.org/rfc/rfc9112.txt

## 02-cybersecurity

建议项目名：`网络安全治理知识图谱`

- `nist-csf-2.0.pdf`：网络安全治理框架、功能、类别、配置文件和成熟度层级。
- `nist-sp-800-61r3-incident-response.pdf`：安全事件、响应流程、组织职责和风险管理活动。

重点观察：Framework、Function、Category、Incident、Organization、Activity、Risk 等实体，以及 contains、supports、responds_to、governs、mitigates 等关系。

来源：

- https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf
- https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-61r3.pdf

## 03-medical-tuberculosis

建议项目名：`结核病诊疗知识图谱`

- `who-tb-care-support.pdf`：结核病治疗、患者照护和支持措施。
- `who-tb-children-adolescents.pdf`：儿童及青少年结核病诊断、治疗和照护建议。

重点观察：Disease、PatientGroup、Drug、Treatment、DiagnosticMethod、Recommendation、Organization 等实体，以及 treats、diagnoses、recommended_for、contraindicated_for 等关系。该组适合测试医学缩写、药品名称和相似诊疗概念的保守归一。

来源：

- https://www.who.int/publications/i/item/9789240047716
- https://www.who.int/publications/i/item/9789240046764

## 04-mineral-supply-chain

建议项目名：`关键矿产产业链知识图谱`

- `usgs-lithium-2025.pdf`：锂的生产、用途、资源、国家和市场数据。
- `usgs-cobalt-2025.pdf`：钴的生产、用途、资源、国家和市场数据。

重点观察：Mineral、Country、Company、Mine、Product、Application、Market 等实体，以及 produces、exports、used_in、located_in、substitutes_for 等关系。两份文档结构相似但矿种不同，适合测试“结构相同、实体不同”的场景。

来源：

- https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-lithium.pdf
- https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-cobalt.pdf

## 05-ai-governance

建议项目名：`人工智能治理知识图谱`

- `nist-ai-rmf-1.0.pdf`：AI 风险管理框架、可信属性、组织角色和治理活动。
- `eu-ai-act-2024.pdf`：AI 系统风险分类、提供者和部署者义务、监管机构与法律要求。

重点观察：AISystem、Risk、Actor、Obligation、Requirement、Authority、Framework、Control 等实体，以及 classifies、requires、applies_to、governs、mitigates 等关系。该组适合验证技术治理框架与法律法规之间的 Schema 扩展。

来源：

- https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689

## 使用提示

- 首轮建议从 `04-mineral-supply-chain` 开始，两份 PDF 都只有一页，处理最快。
- `01-software-protocols` 和 `05-ai-governance` 内容较长，更适合压力测试分块、Schema 采样和实体归一性能。
- 不建议一次把所有文件放入同一项目；这些目录代表不同领域，应分别创建项目，才能验证项目级 Schema 和图谱隔离。
- 测试跨文档归一时，应顺序上传同一目录中的两个文件，而不是同时上传全部文件，以便对比第二次上传前后的变化。
