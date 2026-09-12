# Research Progress Report

**Title:** Boundary-Aware Supervision Amplification for Active Learning in Urban Remote Sensing Classification  

**Student / Project:** gee-project (urban RS active learning)  
**Target venue:** *Remote Sensing* (MDPI)  
**Report date:** August 2026  

---

## 1. Research Goal

提升主动学习（Active Learning, AL）在城市遥感土地覆盖分类中的标注效率。

核心问题不是“再设计一个采样分数”，而是：

> 当高价值样本已被准确发现时，为何分类精度仍难提升？如何把稀缺人工监督有效放大到决策边界区域？

---

## 2. Method Overview

研究形成“发现 → 稀释 → 放大”的完整链条：

1. **BAMS**（Boundary-Aware Misclassification Sampling）  
   优先选择靠近决策边界、更具信息量的疑似误分类样本（相对传统 Margin Sampling）。

2. **Supervision Dilution Effect（监督稀释效应）**  
   少量高价值样本（150 / ~15000，约 <1%）难以直接改变大样本空间的决策边界，因而 OA 不必然提升。

3. **Boundary-Aware Prototype Expansion (PE)**（主方法）  
   以人工标注的 BAMS 样本为原型，在特征空间向低 margin 边界候选传播软伪标签，放大边界监督而非易分内部样本。

4. **Human Amplification (HA)**（消融 / 对照）  
   样本中心式近邻传播；在 TopN 选优协议下可与 PE 接近，但论文主推仍为 **BAMS + Boundary PE**。

---

## 3. Key Scientific Findings

### 3.1 Informative discovery ≠ immediate OA gain

五城 × 五随机种子结果（均值）：

| Method | Mean OA | Mean Boundary Error |
|--------|---------|---------------------|
| Baseline | 91.75% | 256.16 |
| BAMS150 | 91.73% | 257.08 |
| **PE-Boundary** | **92.73%** | **214.76** |
| PE+TS-Boundary | 92.74% | 214.84 |

- BAMS150 相对 Baseline 几乎无增益（甚至略降），支持 **Supervision Dilution**。  
- PE 使 OA 约 **+0.98 pp**，Boundary Error 约 **−16.2%**。  
- PE vs Baseline / BAMS 的配对 t 检验显著（AllCities_Mean）。

### 3.2 主误差来自 Class1↔Class2 边界，而非水体

系统混淆矩阵分析表明：主要误差为 **Class1 ↔ Class2**（建筑边缘、居住区边界、城乡交错带等），而非早期关注的 Class3（水体）。

为此定义：

\[
\text{Boundary Error} = \#(Class1\to Class2) + \#(Class2\to Class1)
\]

实验表明 Boundary Error 下降与 OA 提升同向，形成可解释闭环。

### 3.3 BAMS 确实打中边界（人工验证）

五城共人工核查 100 个 BAMS 样本：

| City | Verified | True Boundary Errors | Hit Rate |
|------|----------|----------------------|----------|
| Wuhan | 20 | 18 | 90% |
| Hefei | 20 | 19 | 95% |
| Nanchang | 20 | 17 | 85% |
| Nanjing | 20 | 19 | 95% |
| Changsha | 20 | 18 | 90% |
| **Total** | **100** | **91** | **91%** |

结论：BAMS 具有较高 **Boundary Error Discovery Precision**；其价值在“找对地方”，而非单独把 OA 抬上去。

### 3.4 Nanjing 作为 Failure / Limitation Case

南京是唯一 PE 相对 Baseline **不显著**、且 HA/BAMS 常无提升甚至变差的城市。

主要原因（已有诊断证据）：

1. **基线最难**：Baseline OA 最低（~90.3%），Boundary Error 最高（~304），Class2 召回最弱。  
2. **人工纠错与评测标签冲突最大**：仅加入 BAMS 人工标签时，相对原始 `Class` 评测，南京 ΔOA≈−0.64 pp、ΔBE≈+23；预测翻转中“伤害 GT”远多于“帮助 GT”。  
3. **可扩展边界池最大**：放大把冲突监督进一步扩散，故 HA/PE 难以表现为 OA 提升。

论文中建议用半页 **Limitations / Failure Case: Nanjing** 说明适用条件，而非回避。

---

## 4. Experimental Setup

- **Cities:** Wuhan, Hefei, Nanchang, Nanjing, Changsha  
- **Candidates:** 每城约 15,000；特征为 3×3 光谱—指数统计量  
- **Query budget:** 150（BAMS / Margin 等）  
- **Seeds:** 0–4（五种子稳定性与显著性检验）  
- **Main metrics:** OA, Kappa, Boundary Error  
- **Main compared methods:** Baseline, BAMS150, PE-Boundary, PE+TS；HA 作协议对齐消融  

HA v2（TopN∈{50,100,150,250} 按 BE 选优，与 PE 的 K 选优协议对齐）后：

| Method | Mean OA | Mean BE |
|--------|---------|---------|
| PE-Boundary | 92.73% | 214.8 |
| HA-Corrected | 92.70% | 215.9 |
| HA-Aligned | 92.67% | 218.4 |

HA 与 PE **差异不再显著**；因此主方法仍写 PE，HA 证明“放大有效，但需门控与选优”。

---

## 5. Current Contributions（拟投稿表述）

1. **提出并验证 Supervision Dilution Effect**：高价值样本发现 ≠ 立即 OA 提升。  
2. **误差归因**：城市遥感主误差为 Class1–Class2 边界混淆；引入 Boundary Error。  
3. **提出 Boundary-Aware Prototype Expansion**：在边界约束下放大人工监督。  
4. **建立可解释链条**：Boundary Error ↓ → OA ↑；并以 Nanjing 说明方法边界条件。

---

## 6. Paper Preparation Status

### 已完成
- 主实验与五种子显著性分析  
- BAMS 人工边界命中验证（91%）  
- Nanjing 失败机制诊断  
- 方法流程图  
- 五城区位示意图（Study Areas + schematic YREB）  
- Wuhan / Changsha / Nanjing 全市 BAMS 点分布截图（GEE）  

### 进行中 / 投稿前建议补齐
1. **可视化收尾**  
   - 五城 BAMS 分布联图  
   - 2–3 张局部边界特写（成功城 + Nanjing）  
2. **（强烈建议）BAMS vs Margin/Random 命中率对照**  
3. **（建议）标准 AL 对照至少一项**（如 Entropy150）  
4. **撰写初稿**（Intro / Method / Experiments / Discussion）  

### 明确不再继续的方向
- 不再开大实验刷 OA  
- 不以“更新全部 ~15000 标签”为主叙事  
- HA 不替代 PE 作为主方法  

---

## 7. Target Journal Assessment（Remote Sensing）

- **匹配度：** 城市土地覆盖、主动学习、标注效率、多城验证——符合期刊范围。  
- **优势：** 问题机制清晰（稀释 + 边界放大），多城×多种子，有人工验证与 failure case。  
- **风险：** OA 绝对增益约 1 个百分点（高基线上）；需靠机制、边界指标、图与对照撑起贡献。  
- **判断：** **可以冲 Remote Sensing，但应以 revision 心态准备**；补齐图与必要对照后，进入认真审稿并有修改机会的概率更高。不建议宣称“稳中”。

---

## 8. Next 2-Week Plan

| Priority | Task |
|----------|------|
| P0 | 完成局部边界截图；整理 Fig 清单 |
| P0 | 写论文骨架（尤其 Intro 问题叙述 + Method PE） |
| P0 | 写入 Nanjing Failure 半页 |
| P1 | BAMS vs Margin/Random 命中率对照表 |
| P1 | Entropy150（或等价）一行对照 |
| P2 | 全稿英文初稿与 Remote Sensing 模板排版 |

---

## 9. One-Paragraph Summary for Advisor

本研究发现：城市遥感主动学习中，BAMS 能以约 91% 命中率找到 Class1–Class2 真实边界错误，但 150 个高价值样本相对约 15,000 候选会被严重稀释，OA 几乎不提升（Supervision Dilution）。据此提出边界感知原型扩展（PE），仅在低 margin 边界区域放大人工监督；五城五种子实验显示 OA 约提升 1 个百分点、Boundary Error 约下降 16%，且多数城市显著。Nanjing 因人工纠错与评测标签冲突及边界最难而成为 failure case。现阶段主实验已完成，正在补齐可视化与必要对照并撰写投 *Remote Sensing* 的初稿。

---

## 10. Materials Available for Discussion

- `data2/overall_seed_summary.csv`, `significance_tests.csv`, `seed_experiments_all.xlsx`  
- `data2/human_amplification/`（HA 规模—精度曲线）  
- `figures/Fig_method_flowchart.png`  
- `figures/gee_shp/*_BAMS150.zip`（GEE 上传用）  
- 五城区位图与 Wuhan/Changsha/Nanjing 分布截图  

如需，我可以再准备一页 **PPT 提纲**（问题—方法—主表—Nanjing—下一步）方便组会汇报。
