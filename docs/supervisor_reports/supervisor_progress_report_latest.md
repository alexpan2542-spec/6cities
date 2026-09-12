# 城市遥感主动学习研究进展报告（最新版）

**对象：** 指导教师  
**日期：** 2026-08-08（更新）  
**课题：** 边界感知主动采样（BAMS）+ 置信度过滤（Confidence）+ 原型扩张（PE）用于城市三分类土地覆被校正  
**评测锚：** ESA WorldCover（WC）三类别（Class1≈建成，Class2≈非建成，Class3≈水体）  
**城市：** 武汉、合肥、南昌、长沙、南京（每城约 15,000 点）  
**拟投：** _Remote Sensing_（MDPI）

相对上一版报告，本版新增：**合肥完成 Confidence 重标与 PE 重跑、大棚种子消融、跨城人–WC 不一致率更新，以及南京–合肥同协议对照叙事。**

---

## 1. 方法主线（未改）

固定标签预算下的三步：

1. **BAMS**：选边界/难例（回答“标哪些”）
2. **Confidence**：只用高置信人标作硬种子（回答“哪些能用”）
3. **PE**：特征空间原型扩张 + 低 margin 软标签（回答“怎么用少而干净的种子涨点”）

核心主张：  
**选点 ≠ 用点 ≠ 扩点。** BAMS 找得到错，不等于全部 150 点直接训练能涨 OA；高置信 + PE 才是主增益来源。同时用南京刻画 **何时失效**。

---

## 2. 五城主结果概览

### 2.1 Phase-1 指纹（历史主表；合肥旧人标时的 PE ≈ +0.87 pp）

| City          | Base OA   | Base BE | BAMS ΔOA     | PE ΔOA（旧设定） |
| ------------- | --------- | ------- | ------------ | ---------------- |
| Wuhan         | 0.910     | 297     | ≈0           | **+1.82 pp**     |
| Changsha      | 0.930     | 234     | ≈0           | **+1.52 pp**     |
| Nanchang      | 0.912     | 206     | ≈0           | **+1.35 pp**     |
| Hefei（旧标） | 0.932     | 240     | ≈0           | **+0.87 pp**     |
| **Nanjing**   | **0.903** | **304** | **−0.27 pp** | **+0.36 pp**     |

南京多项“最差”排名 #1：基线最难、BAMS 净伤害最大、PE 增益最小、低 margin 池最大。

### 2.2 Help / Hurt（WC 评测下）

| City        | BAMS help/hurt        | PE help/hurt     |
| ----------- | --------------------- | ---------------- |
| 多数城      | ≥0.9，PE 常 >1.5      | 明显净帮助       |
| **Nanjing** | **0.66（hurt≫help）** | **1.14（勉强）** |

---

## 3. 合肥更新（Confidence 重标）——本阶段重要进展

### 3.1 标注概况

- 已为合肥 BAMS150 增加 **Confidence**：`h` 81 / `m` 64 / `l` 5
- 相对 backup，Human_Class 改动 **58** 点
- 人–WC 不一致：旧标约 **81%** → 新标约 **49%**（标注更稳，虚高冲突下降）
- 高亮大棚：协议定为 **Class2**（光谱像建成、语义偏农业）；Scene 标注“大棚”

### 3.2 PE 重跑（5 seeds，相对 Baseline）

| Method                   | ΔOA (pp)  | 说明            |
| ------------------------ | --------- | --------------- |
| PE_old（旧人标）         | +0.87     | 与 Phase-1 一致 |
| **PE_new（新全部 150）** | **+1.15** | 重标后略升      |
| **PE_new_high（81）**    | **+1.08** | ≈ 全用          |
| PE_new_hm                | +1.04     |                 |
| 直接写 BAMS（RF）        | ≤0        | 仍无效          |

**结论：** 合肥方法逻辑 **没有变轨**——仍是 BAMS→PE 有效城。Confidence 主要带来：  
（1）与南京 **同协议**；  
（2）证明此处 **不必靠丢掉 med/low 才涨**（high≈all）；  
（3）人–WC 冲突从“极端 81%”回到中等水平。

### 3.3 大棚种子消融

剔除 Scene/Notes 含「大棚/温室」的 **9** 个种子（high 中 7 个）：

| Method        | 种子数 | ΔOA   |
| ------------- | ------ | ----- |
| PE_all        | 150    | +1.15 |
| PE_all_no_gh  | 141    | +1.01 |
| PE_high       | 81     | +1.08 |
| PE_high_no_gh | 74     | +0.99 |

去掉大棚约 **−0.1 pp**，可忽略。合肥增益不是靠大棚种子；大棚标 2 也未破坏主结果。

---

## 4. 南京现状（failure case，证据更完整）

### 4.1 Confidence + PE

| ceMethod          | ΔOA            |
| ----------------- | -------------- |
| PE 全人标         | ~+0.21         |
| **PE_high（73）** | **~+0.33**     |
| PE_high+med       | ~+0.17（更差） |

→ 南京存在 **supervision dilution**：必须滤置信；但滤完仍只有约 +0.3 pp。

### 4.2 Round2（+50 个 C1–C2 软边界）

- high/med/low = 33/14/3；相对 WC 翻转 **74%**
- PE 仅 Round2 high：**+0.40 pp**
- 与 BAMS high 合并：不再提升；并入 med 略降

→ 针对性高置信边界种子有效，但堆数量/合并 **冲不破天花板**。

### 4.3 双轨评测（WC-OA vs Human-OA，严格 hold-out）

BAMS high ∪ Round2 high = 106 点；60% 种子 / 40% 人标测试。  
人标 hold-out 与 WC 一致率仅 **~35%**。

| Method              | WC-OA   | Human-OA             |
| ------------------- | ------- | -------------------- |
| Baseline MLP        | 0.909   | 0.665                |
| PE hold-out（正当） | 略降    | **+2.3 pp**          |
| 按 WC 选 K          | WC 稍好 | Human 变差           |
| 泄漏对照            | —       | Human 虚高（不可用） |

→ 换人标评测也不能 magically 翻盘；两金标准互相打架。

### 4.4 High × 3×3 邻域

73 个 high 与 WC 邻域 `suggest` 仅 **38%** 一致；只保留“人=suggest”的 28 点做 PE 时 WC-OA 略好（+0.43 vs +0.33）。  
邻域可作过滤器，不可当真理改高置信人标。

### 4.5 High-only teacher → student

经典 TS、PE+TS、或 teacher **只看 73 high**，均 **不优于 PE_high**，无法把南京拉到 +1 pp。

---

## 5. 跨城人–WC 不一致（最新）

同一 BAMS150 协议，`Human_Class ≠ WC`：

| cCity       | 不一致率        | Confidence                         |
| ----------- | --------------- | ---------------------------------- |
| **Nanjing** | **68.0%（#1）** | 有；high 子集 61.6%                |
| Wuhan       | 58.0%           | 无（旧标）                         |
| Changsha    | 52.7%           | 无                                 |
| **Hefei**   | **49.3%**       | 有；high 子集 53.1%（旧标曾 ~81%） |
| Nanchang    | 44.7%           | 无                                 |

**合肥重标后，南京成为当前表上人–WC 最不一致的城市。**  
（武/长/昌若也重标，排名可能再变。）

### 5.1 能否说“南京考卷不一样所以 OA 低”？

- **能说：** 五城都用 WC 阅卷，但南京边界上「改卷（人）」与「阅卷（WC）」错位最大，会压低可见的 WC-OA 增益。
- **不能说：** 南京换了另一套产品当考卷；或“不一致高 ⇒ 必然失败”（合肥 ~49% 仍 +1.1 pp）。
- **完整因果：** **参考冲突最大 × 基线最难 × BAMS 在 WC 上净伤害** → PE 增益塌缩。

---

## 6. 合肥 vs 南京：同协议对照（叙事核心）

| 维度           | Hefei（新 Confidence） | Nanjing           |
| -------------- | ---------------------- | ----------------- |
| PE_high        | **~+1.08～1.15 pp**    | **~+0.33 pp**     |
| high vs 全用   | **几乎相同**           | **high 明显更好** |
| 人–WC（BAMS）  | 49%                    | **68%（最高）**   |
| 基线           | 易（OA~0.93）          | 最难              |
| BAMS help/hurt | 接近中性               | hurt≫help         |
| 大棚等噪声     | 不敏感                 | —                 |
| 结论           | **成功案例**           | **failure case**  |

可写进论文的对照句：

> 两城均采用 Confidence 过滤。合肥 high-only 与全量 PE 接近，增益约 +1 pp，说明链路有效。南京同样只用 high，增益仍约 +0.3 pp，且人–WC 不一致最高、BAMS 在 WC 评测下净伤害——失败归因于参考冲突与场景难度耦合，而非“未做置信过滤”。

---

## 7. 当前贡献表述（建议）

1. **BAMS** 对准城市建成边界错误。
2. **Supervision dilution**：找错 ≠ 全用；需 Confidence。
3. **PE + high** 在多数城市带来约 **+1 pp** OA、BE 下降。
4. **适用边界**：南京证明当 WC 边界参考与人标严重冲突且城很难时，同协议下增益受限。
5. 合肥 Confidence 重标使成功/失败对照 **协议一致**，并削弱“合肥冲突 81% 却成功”的审稿攻击点。

---

## 8. 投稿判断与下一步

### 8.1 现状

- 匹配 _Remote Sensing_；主结果约 +1 pp：**可冲、不算稳中**。
- 南京 failure + 合肥同协议对照：Discussion 明显加强。
- 缺口：武/长/昌尚未 Confidence；缺第二参考图（DW/ESRI）钉死 reference bias；缺更强主动学习基线表。

### 8.2 优先下一步

1. 武汉/长沙/南昌补 Confidence（统一五城消融：`PE_all` vs `PE_high`）
2. 五城 BAMS × WC × Dynamic World 一致率表
3. 公平基线（random / margin / entropy）+ 显著性

---

## 9. 给老师的三句话（最新）

1. **方法：** BAMS 选点 → Confidence 用点 → PE 扩点；主增益在 PE，不在把 150 点全写进 RF。
2. **合肥：** 加上 Confidence 后逻辑不变、增益略升到约 +1.1 pp；high≈全用；大棚标 2 且去掉几乎无影响；人–WC 冲突降到约 49%。
3. **南京：** 现为五城人–WC 最不一致；同协议下 PE_high 仍仅约 +0.3 pp——失败故事比以前更好讲：不是没滤置信，而是 **阅卷（WC）与改卷（人）错位最大 + 城最难**。

---

## 附录：关键路径

| 内容               | 路径                                                 |
| ------------------ | ---------------------------------------------------- |
| 合肥 Confidence PE | `data2/hefei_relabel_bams_pe/`                       |
| 合肥大棚消融       | `data2/hefei_greenhouse_ablation/`                   |
| 南京 Confidence PE | `data2/nanjing_relabel_bams_pe/`                     |
| 南京 Round2        | `data2/nanjing_round2_c12_pe/`                       |
| 双轨 WC/Human      | `data2/nanjing_dual_eval_holdout/`                   |
| High×3×3           | `data2/nanjing_high_3x3nb/`                          |
| Phase-1 五城指纹   | `data2/nanjing_phase1_diagnostics/`                  |
| 合肥人标           | `data2/Hefei/04_manual/Hefei_BAMS150_Manual.csv`     |
| 南京人标           | `data2/Nanjing/04_manual/Nanjing_BAMS150_Manual.csv` |
| 上一版报告         | `docs/supervisor_progress_report_2026-08-08.md`      |

---

_数值均为多种子均值；Phase-1 中合肥 PE 为旧人标结果，请以 §3 重跑为准。_
