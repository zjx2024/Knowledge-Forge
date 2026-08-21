# 产业链0724：两次实体抽取与归一结果对比

生成时间：2026-07-27<br>
项目：`产业链0724` / project_id `13`<br>
文档：`usgs-lithium-2025.pdf`、`usgs-cobalt-2025.pdf`<br>
模型配置：`qwen3.6:latest`，`model_max_tokens=262144`，embedding `bge-m3:latest`

本文对比的是当前页面中刚运行过的两套构建结果：

1. `stable` 稳定基线；
2. `enhanced_shadow` 增强方案，即 Shadow v2 关系运行时 + P2 实体归一。

注意：这里没有人工 Gold 标注，因此本文只总结结构性结果和可观察差异，不宣称 precision、recall 或综合质量提升。

## 总览

| 指标 | stable 稳定基线 | enhanced_shadow 增强方案 |
| --- | ---: | ---: |
| 状态 | ready | ready |
| 文档数 | 2 | 2 |
| Chunk 数 | 15 | 15 |
| 实体提及数 | 173 | 337 |
| 规范实体数 | 169 | 265 |
| 归一压缩率 | 2.31% | 21.36% |
| 多提及合并后的规范实体数 | 4 | 70 |
| 最大单实体提及数 | 2 | 4 |
| 平均每规范实体提及数 | 1.02 | 1.27 |
| 当前物化关系数 | 0 | 33 |

归一压缩率按 `1 - 规范实体数 / 实体提及数` 计算。

## 按文档统计

| 文档 | stable 提及数 | stable 规范实体数 | enhanced 提及数 | enhanced 规范实体数 |
| --- | ---: | ---: | ---: | ---: |
| `usgs-cobalt-2025.pdf` | 66 | 65 | 152 | 136 |
| `usgs-lithium-2025.pdf` | 107 | 106 | 185 | 150 |

增强方案的实体提及覆盖明显更高：两份文档共多抽取 164 个提及，规范实体多 96 个。稳定基线几乎是“一提及一实体”，归一非常保守；增强方案则产生更多跨提及合并，说明 P2 归一链路确实在工作。

## 按实体类型统计

### stable 稳定基线

| 类型 | 提及数 | 规范实体数 |
| --- | ---: | ---: |
| Product | 51 | 51 |
| GenericEntity | 46 | 43 |
| Location | 42 | 42 |
| Organization | 19 | 19 |
| Event | 11 | 10 |
| Person | 2 | 2 |
| Technology | 2 | 2 |

### enhanced_shadow 增强方案

| 类型 | 提及数 | 规范实体数 |
| --- | ---: | ---: |
| Location | 128 | 90 |
| Product | 106 | 94 |
| GenericEntity | 41 | 37 |
| Organization | 22 | 18 |
| Technology | 21 | 16 |
| Event | 17 | 13 |
| Person | 2 | 1 |

增强方案在 `Location`、`Product`、`Technology` 上的覆盖提升最明显。尤其是 `Location`，提及数从 42 增至 128，同时被归一到 90 个规范实体，说明它既抽得更多，也确实合并了一部分重复地名。

## 归一行为观察

stable 的合并非常少，只有 4 个规范实体承载了多个提及：

| 规范实体 | 类型 | 提及数 |
| --- | --- | ---: |
| Apparent Consumption | GenericEntity | 2 |
| Depletion Allowance | GenericEntity | 2 |
| Government Stockpile | GenericEntity | 2 |
| worldwide lithium production | Event | 2 |

enhanced_shadow 有 70 个规范实体承载多个提及。部分代表例子：

| 规范实体 | 类型 | 提及数 |
| --- | --- | ---: |
| lithium carbonate | Product | 4 |
| 12–31–24 | Event | 2 |
| 2024e | Event | 2 |
| Alaska | Location | 2 |
| Argentina | Location | 2 |
| Australia | Location | 2 |
| Bolivia | Location | 2 |
| Brazil | Location | 2 |
| Brian W. Jaskula | Person | 2 |
| California | Location | 2 |

这说明增强方案的实体归一不再只是局部去重，而是在跨 chunk、跨文档的同名或别名归一上更积极。

## 风险点与需要复核的地方

增强方案的归一压缩率更高，但也带来更高的误合并风险。当前看到一个值得人工复核的例子：

- `lithium carbonate` 的 alias 中出现了 `lithiumchloride`。

这可能是错误合并，或者是抽取阶段把相邻/相关化合物放进了同一实体别名集合。没有人工 Gold 前，不能把“合并更多”直接等同于“质量更好”。建议后续用小规模 Gold 对这些高影响 Product 合并做人工核查。

## Shadow v2 关系侧附带结果

虽然本文重点是实体抽取和归一，但本次 enhanced 运行也完成了 Shadow v2 关系链路：

| 指标 | 数值 |
| --- | ---: |
| pair opportunities 总数 | 4934 |
| 实际进入分类的 pair | 746 |
| classifier `NO_RELATION` | 691 |
| classifier 产生 hypothesis 的决策 | 55 |
| classified hypotheses | 105 |
| verification attempts | 105 |
| verifier SUPPORTED | 33 |
| verifier REJECTED | 47 |
| verifier UNCERTAIN | 25 |
| normalized facts | 33 |
| legacy pipeline calls | 0 |

状态契约符合预期：只有 verifier `SUPPORTED` 的 hypothesis 才进入 normalized fact；当前 33 条可物化关系均来自 SUPPORTED verification。

## 与历史 enhanced 快照的参考对比

数据库中还保留了 2026-07-24 的旧 enhanced 快照，统计为：

| 快照 | 提及数 | 规范实体数 | 归一压缩率 | 关系端点数 |
| --- | ---: | ---: | ---: | ---: |
| 2026-07-24 legacy enhanced | 335 | 262 | 21.79% | 164 |
| 2026-07-27 native Shadow v2 enhanced | 337 | 265 | 21.36% | 33 |

实体侧两个 enhanced 快照非常接近：提及数和规范实体数只小幅变化，说明 P2 实体抽取/归一总体稳定。关系侧差异很大，是因为 2026-07-27 的 native Shadow v2 执行了独立 verifier 和 fail-closed 发布约束，未通过验证或不确定的关系不会进入可物化结果。

## 结论

1. stable 基线实体归一极保守，173 个提及最终得到 169 个规范实体，几乎不合并。
2. enhanced_shadow 明显提高实体提及覆盖，337 个提及归一为 265 个规范实体，并产生 70 个多提及规范实体。
3. enhanced_shadow 的归一链路确实生效，尤其在地名、产品、技术实体上覆盖更充分。
4. enhanced_shadow 也需要人工 Gold 或抽样复核来评估误合并，当前 `lithium carbonate` / `lithiumchloride` 的别名现象建议列为优先复核案例。
5. 当前只能说 enhanced_shadow 的结构契约和归一覆盖优于 stable，不能在没有 Gold 的情况下声明实体抽取准确率或召回率提升。
