# 论文单一事实源 — 城市土地覆被边界一致性诊断 → MDPI *Remote Sensing*

> **本文件是这篇论文的唯一权威说明。任何时候只读本文件即可从零恢复全部上下文并继续工作。**
> 最后更新: 2026-09-07（Claude 协助整理）。修改本文件时同步更新顶部日期。

---

## 0. 如何随时继续（RESUME 协议）

1. 读完本文件 §1–§4，就掌握了：定调、决策、环境、全部已冻结的实验数字。
2. 要复跑实验：按 §2 的命令。脚本写进 `data/…`，**结果快照另存在
   `docs/results_snapshot/`**（见 §5），复跑不会覆盖快照。
3. 要继续推进：看 §6 待办清单，按优先级挑一项，每项都写清了「做什么/输入/产出」。
4. 写作：按 §3 的 IMRaD 结构；每个 Results 小节对应 §4 的一张表 + 一张图。
5. 所有一致率数字**必须**带 §4.0 的限定语（边界候选点，非全图精度），否则误导。

---

## 1. 现状与决策日志

### 1.1 一句话现状

目标期刊 **MDPI Remote Sensing**（research article）。论文定调：
**诊断为主 + LOCO 校正算子为辅**。诊断核心 = 用 6 城 × 150 个作者逐点手标的
城市建成↔植被边界样本，量化 ESA WorldCover / Dynamic World / ESRI 三个全球
产品与专家判断的系统性分歧，并证明「加第二、第三个产品并不能仲裁这个分歧」。

**实验/数据侧已基本合格**（诊断核心 + LOCO 算子 + 可视化地图都跑通、防泄漏、
已冻结）。**成文进度（2026-09-07）**：`docs/manuscript.md` 已有
§3 Methods（成稿）+ §4 Results（成稿，对齐 6 城冻结快照 n=900——T1/T2/T3/T4/T5
+ 独立性 S2/S3 + Phase 1/2/3 + 前后地图）。**未写**：§1 Introduction、
§2 Study Area & Data、§5 Discussion、§6 Conclusion、Related Work、Abstract。

**明天的起手（顺序）**：
1. 修 `manuscript.md` §3.1 的 "built-up footprint" → GAUL L2 市域（见 §3 大纲的 ⚠️）；
   把 §7.4 的 shared-S2 防守 4 点并进 §3.1 "Sampling-induced conditioning" 段 + §5。
2. 写 §2 Study Area & Data —— 技术事实已锁定（见 §3 大纲「§2 已锁定的技术事实」），
   AOI 正式引用 / 标注者背景表述 / 选点动机政策引用 三处留 `[TODO]` 等用户补。
3. 写 §1 Introduction + Related Work（桶 A/B 结构见 §6.4，引用留占位）。
4. 写 §5 Discussion —— 机制软处理框架已定（2026-09-07 决策日志），素材见 §5 大纲 + §7。
5. 收尾：§6 Conclusion、Abstract、所有表/图 caption、过 RS checklist。

**用户侧未决**（见 §6.4 / §6.6 / §1.3）：三处真实引用；更正 2026-08-08 导师报告；
可选的 13 个南昌场景点视觉核 + 30 个高置信点亚米影像 robustness check。

### 1.2 决策日志（不要推翻，除非有新证据）

| 日期 | 决策 | 依据 |
|---|---|---|
| 2026-09-02 | **Prototype Expansion (PE) 方法作废**，不再作为论文方法 | 结构性测试集泄漏；防泄漏后 `n_expanded=0`，近邻传播实测贡献为零（§4.5） |
| 2026-09-03 | 论文重定调为**诊断优先**，LOCO 算子降为支撑章节 | 算子有真信号但影响适度：只追回本地标注天花板的 ~1/4（§4.4） |
| 2026-09-03 | pipeline **全程用 RandomForest**（基线与所有校正变体同一分类器） | 旧 PE 的「提升」大半来自 RF→MLP 换分类器，必须消除这个混淆 |
| 2026-09-03 | LOCO 算子用**留一城交叉验证 (LOCO)** | 构造上防泄漏：校正器永不见目标城；分类器永不见目标城的人标 |
| 2026-09-05 | 六城选点动机写成**区域实际监测需求**（长江中下游城市群城市化速度快、生态红线/耕地保护/水域岸线管控等政策对土地覆盖动态监测需求大），**不写**「作者在南昌工作、便于开展」这类个人便利理由 | 便利抽样理由会被审稿人读成选点无代表性逻辑，坐实地理泛化局限（§7.7 待补） |
| 2026-09-05 | 标注可信度补充论据：标注者对研究区域有实地遥感地物分类从业经验 | 缓解「Human 只是单一专家判断」这一弱点（§7.4），写进 Methods 标注流程部分，不写进选点动机 |
| 2026-09-06 | **场景代码映射更正**：标注者提供原始代码表后核实，SB(特殊建筑)/BS(建筑阴影)
早期分组脚本搞反了；UB(城市绿化,42点)被错误并入"Built-up edge"；BR/WR/PF 重新归类。
"Soft boundary / shadow" 从 26 点/65.4% 缩水成 3 点/0%，跌出主表；新增"Urban green
space"类别(42点,69.0%，与Built-up edge并列第一)。已重跑 T4，headline 结论不变
（Built-up edge 仍最高），但"Soft boundary/shadow"这个类别不能再用 | 用户提供的
原始场景代码表（WE/TW/WL/CT/CE/PF/GH/BE/RS/RE/BR/BL/FE/BS/UB/SB/CS/MX 共18码）
与之前脚本的猜测式映射有实质出入，涉及 76 个点的归类 |
| 2026-09-06 | 第一次 inter-annotator 复核数据判定无效（同事实为审核已有答案，非独立盲标，Scene/Confidence 字段与原文件 150/150 一致）作废，重新发起严格盲标协议（见 §6.8） | 技术核查发现 Scene/Confidence 全字段一致，独立标注不可能产生这种一致性 |
| 2026-09-05 | **正面写出 PE 弯路**，不回避：Methods 3.3 明确讲「为什么 LOCO 用留一城交叉验证、而非同城内近邻传播」，点名引用 PRE (arXiv 2406.00891 / JPRS 2025)，把「同城内原型/近邻传播在空间自相关下会结构性泄漏」讲成一个方法论论证，而非隐瞒的失败史 | 不点名会被懂行审稿人读成回避在先文献（PRE 是同类方法家族的代表作）；正面写出来反而是可信的方法论贡献——证明 LOCO 的设计选择是有依据的排除法，不是随意选的 |
| 2026-09-07 | **DW/ESRI 耦合的机制解释 = 软处理**：Discussion 单开一小节、标题带 "hypothesis"，先给纯统计结论（错误 Yule's Q ≈ 0.97、excess +0.25 保守下界、仲裁 PPV 0.43 ≈ 单产品 0.44），再把候选解释**并列**（深度分割方法族 / 共享架构先验 / 共享预处理 / 点标注语料谱系），不排强弱序。「方法族差异」（WC = boosted-tree+规则 vs DW/ESRI = 深度语义分割）可用公开文档说死并引用；「共享标注语料谱系」重度 hedge。收尾：证实需产品内部信息、非公开 → future work | 统计结论本身不需机制即成立（论文脊梁）；一旦断言「共享训练数据」会招来懂产品的审稿人要拿不出的证据，能翻船。写成「可检验但公开文档无法证实的假设」反而显成熟 |
| 2026-09-07 | **LOCO 不做加强**（plan §6.5 降级为「不做」）：不加新实验，§4.6 Phase 1/2/3 结果保持原样。只调 framing——§4.6 收尾 + §5 写清 LOCO 是回答诊断提出的问题「这个分歧是否结构化到能被自动校正」，答案「信号跨城迁移、下游影响有界」本身是诊断结论，非产品推销；LOCO 全节 ≤ 1.5 页、明确从属 | 0.52→0.55 不改变故事却费时；做大 LOCO 会被按「方法贡献」审（要 baselines/ablation/与其它迁移法比），攻击面骤增。保持小而诚实，攻击面就小 |
| 2026-09-07 | **标注主判据 = 2021 S2 合成**（真彩 B4/B3/B2 + 假彩 B8/B4/B3），Google Earth HYBRID 高分底图仅作辅助空间上下文、S2 不足时切用；按 10 m 像元内优势地物定类 | 三产品都从 ~10 m S2 产出，用同尺度、时相对齐（2021）的参考才公平；重度依赖亚米底图会在比产品能达到的分辨率更细的尺度上打标签、不公平放大产品误差。**代价**：参考与选点共享 S2 信息基 → 需在 §3.1/§5 写防守（见 §7.4 补充 4 点） |

### 1.3 待更正的历史材料

给导师的 `docs/supervisor_progress_report_latest.md`（2026-08-08）仍写
「主增益在 PE」。**需主动更正**，说明是自查发现的结构性泄漏。独立于写作，越早越好。

---

## 2. 环境与复现

**Python 环境**: conda env `gee`
`/opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3`
（numpy 2.4 / pandas 3.0 / sklearn 1.9 / scipy 1.17 / matplotlib 3.11；**无 torch**，
pipeline 是纯 RandomForest，刻意为之）。项目自带的 `.python_pkgs/` 只有 torch，
系统 `python3` 无 sklearn——**必须用上面这个 conda python**。

```bash
cd /Users/alex/Projects/gee-project
P=/opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3

# 诊断分析（论文核心表 T1–T5 + 4 张图）—— 秒级
$P scripts/diagnostic_analysis.py
#   -> data/analysis_outputs/diagnostic_analysis/{T1..T5}.csv, SUMMARY.txt
#   -> figures/Fig_diag_{fourway,confidence,scene,error_direction}.png

# 独立性诊断（把「DW/ESRI 互相像却都不像专家」形式化 + 排混淆项）—— 秒级
$P scripts/independence_diagnostic.py
#   -> data/analysis_outputs/independence_diagnostic/{S2a,S2b,S2c,S3a,S3b,S3c}.csv, SUMMARY.txt
#   -> figures/Fig_diag_independence.png

# LOCO 算子（Phase1 迁移 / Phase2 下游 / Phase2b sweep / Phase3 裁判）—— 约 8–12 min
$P scripts/loco_correction_operator.py
#   -> data/analysis_outputs/loco_correction_operator/{phase1_corrector_transfer,phase2_downstream,
#      phase2_by_city,phase2_summary,phase2b_sweep_summary,phase3_referee_eval,
#      phase3_referee_summary,phase3_reference_divergence}.csv, run.log

# 边界修正前/后可视化地图（全 6 城）—— 秒级，复用 loco 算子
$P scripts/make_correction_maps.py
#   -> data/analysis_outputs/correction_maps/Fig_diag_map_{City}.png  (全 6 城)

# 泄漏诊断（历史，证明 PE 作废的依据）—— 在 gee-project2 里
/Users/alex/Projects/gee-project2/... spatial_block_validation.py   # 见 §5
```

**`data/` 目录结构（2026-09-06 重组后）**
- `data/cities/{City}/` — 6 城主数据（01_original / 02_margin / 04_manual / 06_3x3 等）
- `data/shared_reference/` — 跨城参考：`cross_city_second_ref_dw_esri/`（DW/ESRI）、
  `cross_city_scene_localization/`（场景表）、`cross_city_aoi_season_balance/`
- `data/analysis_outputs/` — 当前脚本输出：`diagnostic_analysis/`、
  `independence_diagnostic/`、`loco_correction_operator/`
- `data/nanjing_rootcause/` — `nanjing_baseline_autopsy/` + `nanjing_rootcause_v2/`（Discussion 用）
- `data/interannotator_recheck/` — 标注盲标复核（§6.8）
- 旧 PE / 探索性实验的脚本与 `data/` 产物已全部清理，见 `docs/CLEANUP_LOG.md`。

**关键脚本**
- `scripts/diagnostic_analysis.py` — 诊断核心。读 6 城 manual + 01_original 的 WC +
  `shared_reference/cross_city_second_ref_dw_esri/` 的 DW/ESRI +
  `shared_reference/cross_city_scene_localization/` 的场景表。
- `scripts/independence_diagnostic.py` — 独立性诊断（§4.1b）。复用上面的 loader；
  把 4.1 的「0.87 vs 0.42」升级成条件独立性假设的可检验违背 + 排除类别体系/共享输入/
  时相三个混淆项。
- `scripts/loco_correction_operator.py` — LOCO 算子，含完整 docstring 说明三 Phase 设计。
- `scripts/make_correction_maps.py` — 复用 loco 模块的算子和 gate，出 §4.7 的三联可视化地图。
- `../gee-project2/scripts/spatial_block_validation.py` — 泄漏诊断 + 空间分块重验证。

---

## 3. 论文结构（IMRaD，对应 Remote Sensing 常见格式）

1. **Introduction** — 全球土地覆被产品在城市建成/植被边界的系统性误标；
   「多个全球产品能否互相仲裁边界真值」这一问题此前未被正面回答；
   在极少人工预算下如何做这个诊断。
2. **Study Area & Data** — 六城（武汉/合肥/南昌/南京/长沙/杭州）；Sentinel-2
   6 波段 + NDVI/NDBI/MNDWI + 3×3 邻域均值/标准差；WC / DW / ESRI 三产品；
   150 点/城边界样本的手标流程（Confidence 三级 + Scene 文本协议）。
   **明确交代：BAMS 采样使样本富集于难点，所有一致率是边界候选点上的、非全图精度。**
   **选点动机**（2026-09-05 定稿，见决策日志）：写成长江中下游城市群城市化速度快、
   生态红线/耕地保护/水域岸线管控等政策对土地覆盖动态监测需求大——**不写**作者
   个人便利理由。⚠️ 「城市化速度最快/政策需求最大」这个断言写作时**必须配一个
   真实引用**（城市化统计年鉴、长江经济带生态环境政策文件、或相关文献），
   不能空口断言，否则审稿人会要证据。
   标注流程部分补一句：标注者对研究区域有实地遥感地物分类从业经验，增强参考
   标签可信度（缓解 §7.4「Human 只是单一专家」的弱点）。

   **§2 已锁定的技术事实（2026-09-07 从 `archive/pe_era_scripts/hangzhou_init_pipeline.py` +
   杭州标注脚本核出，6 城一致——docstring 明写 "Matches existing city protocol"，
   杭州只有 BAMS 排序公式不同，见 §3.1）：**
   - **研究范围 (AOI)**：每城 = `FAO/GAUL/2015/level2` 的 ADM2 多边形（地级市
     **全市域**；ADM0='China' + ADM1=省 + ADM2=市），`fc.geometry()` 整块，
     **没有再套建成区掩膜**。
   - **撒点**：ESA WorldCover v200（`.first().select("Map")`）remap 成 3 类
     （WC_FROM=[10..100] → WC_TO=[2,2,2,2,1,2,2,3,3,3,2]，即 50→建成、
     80/90/95→水、其余→非建成），`stratifiedSample(numPoints=5000,
     classBand="Class", region=roi, scale=10, seed=42, classValues=[1,2,3],
     classPoints=[5000,5000,5000])` → 15000 点 / 城，全市域内按类分层。
   - **S2 合成**：`COPERNICUS/S2_SR_HARMONIZED`，`filterBounds(roi)` +
     2021-01-01..2021-12-31 + `CLOUDY_PIXEL_PERCENTAGE < 20` + `.median()` +
     `.clip(roi)`；B2/3/4/8/11/12 + NDVI=(B8−B4)/(B8+B4)、NDBI=(B11−B8)/(B11+B8)、
     MNDWI=(B3−B11)/(B3+B11)。与 BAMS 特征用的是同一张合成。
   - **标注平台/底图**：GEE Code Editor 自写脚本逐点（PointID 1..150，
     `Map.centerObject(point, 16)`）。**主判据 = 2021 S2 合成**：真彩 B4/B3/B2
     拉伸 0–3000，假彩 B8/B4/B3 拉伸 0–4000（默认关，需要时开）。
     `Map.setOptions('HYBRID')` 的 Google 高分卫星底图**仅作辅助空间上下文**、
     S2 不足时切用（2026-09-07 决策日志）。按 10 m 像元内**优势地物**定类，
     Confidence(H/M/L) 记可判读性。
   - ⚠️ **Caveat 要写**：Google HYBRID 底图无统一成像日期、各地新旧不一（可能差
     数年）；判读以 2021 S2 为时相基准，高分底图仅用于地物形态/边界定位。
   - ⚠️ **待你补的 §2 事实**：AOI 引用（GAUL / FAO 2015 的正式 cite）；标注者
     背景的具体表述（哪类地物 / 多久 / 什么项目）。选点动机的真实政策/统计引用
     （见上）。
3. **Methods**
   - 3.1 BAMS 边界候选采样：margin（RF 概率 top1−top2）+ 3×3 局部光谱标准差
     ⚠️ **`docs/manuscript.md` §3.1 有一处待修（2026-09-07 发现）**：现写
     "drawn ... over the **built-up footprint** of the city" —— 错。实际是在
     **整个 GAUL Level 2 市域**内按 WC 类分层撒点，无建成区掩膜。把样本拉到
     地物过渡带的是 **BAMS 那一步**，不是空间掩膜（也解释了 T4 里为何有
     paddy / forest edge / rural settlement）。改法：pool = GAUL L2 市域 +
     WC 类分层；BAMS = 富集机制。同时把 §7.4 补的 shared-S2 防守 4 点并进
     §3.1 的 "Sampling-induced conditioning" 段 + §5。
   - 3.2 多参考一致性诊断协议：四方一致 / 置信度分层 / WC 误差方向 / 场景交叉；
     Wilson 95% CI
   - 3.3 LOCO 校正算子（支撑）：其他 N−1 城的手标点 → RF（27 特征，置信度加权）
     → 迁移到留出城；boundary-gate（低 margin）+ confidence-gate（RF proba ≥ τ）。
     **设计动机段落（2026-09-05 定稿，正面写出弯路，不回避）**：先点名 PRE
     (Tong et al., arXiv 2406.00891 / JPRS 2025) 这类同域内原型/伪标签扩张方法，
     再讲清楚我们最初按此思路做过一版同城内近邻传播（即已作废的 Prototype
     Expansion, PE），实测在城市内部空间自相关下产生结构性标签泄漏（防泄漏后
     `n_expanded=0`，见 §4.5）——因此改为跨城训练、目标城完全不参与训练的
     留一城交叉验证设计。这段写成「排除法得出的设计依据」，不写成「失败了
     所以放弃」的叙事。
   - 3.4 评价协议：留一城交叉验证；下游实验用空间分块（6×6 网格按 lat/lon 整块划）；
     随机翻转对照
4. **Results** — 见 §4，每小节一表一图
   - 4.1 四方一致性 → 表 T1 + `Fig_diag_fourway.png`
   - 4.2 置信度分层 → 表 T2 + `Fig_diag_confidence.png`
   - 4.3 WC 误差方向 → 表 T3 + `Fig_diag_error_direction.png`
   - 4.4 场景定位 → 表 T4 + `Fig_diag_scene.png`
   - 4.5 LOCO 算子迁移与裁判评测 → 表 T5（诚实报告局限）
   - 4.6 边界修正前/后可视化地图（全 6 城；主文 Nanjing+Changsha，其余补充材料）→ 见 §4.7（✅ 已完成）
5. **Discussion** — DW/ESRI 为何互相像却都不像专家：**先给结果（§4.1b 已证条件
   独立性假设被违背、仲裁零增益，这是纯统计结论）**，再把「共享深度分割方法族 +
   S2 点标注语料谱系」作为**解释假设**列出（并列其他候选：共享架构先验、共享
   预处理；类别体系/共享输入/时相三项已在 §4.1b 排除）——用公开文档（DW: Brown
   et al. 2022；ESRI/IO: Karra et al. 2021；WC: ESA ATBD）支撑「方法族」层面的
   区分，不宣称「同一批训练数据」这种无法证实的强断言；
   南京 root-cause（C2 类同时混进建成与水体，见
   `data/nanjing_rootcause/nanjing_baseline_autopsy/VERDICT.md`）；跨城迁移只追回 1/4 天花板的含义；
   置信度分层的抽样偏（见 §7）；局限性。
6. **Conclusion**

---

## 4. 冻结的实验结果（可直接进表；快照见 `docs/results_snapshot/`）

### 4.0 所有数字的强制限定语

以下「一致率 / 分歧率」都是在 **BAMS 挑出的边界候选点**（每城约 15000 点里
最难的一批）上测的，**不是 WorldCover / DW / ESRI 的全图精度**。文中每处都要
写成「在边界候选点上」。类别编码：1/2/3 =（建成 / 非建成 / 水体），代码内部 −1 → 0/1/2。
Boundary Error (BE) = 混淆矩阵里 C1↔C2 的对称混淆计数 `cm[0,1]+cm[1,0]`。

### 4.1 四方一致性 (T1) — `docs/results_snapshot/diagnostic_analysis/T1_fourway_agreement.csv`

5 城（有 DW+ESRI，无杭州）合并，边界候选点 n = 750：

| 对比 | 一致率 |
|---|---|
| Human = WorldCover | **0.46** |
| Human = Dynamic World | 0.43 |
| Human = ESRI | 0.42 |
| WC = DW | 0.46 |
| WC = ESRI | 0.48 |
| **DW = ESRI** | **0.87** |
| 三方 (H, DW, ESRI) 全一致 | 0.37 |
| 四方全一致 | **0.16** |

逐城 Human=WC：Wuhan 0.41 / Hefei 0.51 / Nanchang 0.55 / **Nanjing 0.32（最低）** /
Changsha 0.51 / Hangzhou 0.53。6 城合并 Human=WC 0.47（n=900）。

**叙事句**：DW 与 ESRI 彼此高度一致（0.87），但两者与专家都只有约 0.42；
换第二、第三个全球产品当参考并不能仲裁城市边界——四方在约 2/3 的边界点上没有
共识，可仲裁的只有「三方全一致」这个约 37% 的子集。**机制层面的定位见 §4.1b：
本文把这句话形式化为「条件独立性假设的可检验违背」，而不把「共享训练先验」
当结论下——那是 Discussion 里的解释假设之一（§5、§7.4）。**

### 4.1b 独立性诊断 (S2/S3) — `docs/results_snapshot/independence_diagnostic/`

> **写作定位（2026-09-06）**：4.1 的「0.87 vs 0.42」本身只是现象。多参考仲裁
> 默认「多个参考相互独立地逼近真值，一致即可信」。本节直接检验这个前提，
> 结论是它在城市边界像元上不成立——**这一步不需要机制**，是纯统计结果。
> 「为何不独立」（共享深度分割方法族 / 点标注语料谱系）降级为 §5 的解释假设。
> 5 城、n = 750。

**S2a — 观测一致率 vs「给定专家标签条件独立」时的期望一致率**

| 对比 | 观测一致率 | 条件独立期望 | 超出量 excess | bootstrap 95% CI | 留一城范围 |
|---|---|---|---|---|---|
| **DW = ESRI** | 0.869 | 0.628 | **+0.241** | [0.21, 0.27] | [0.23, 0.25] |
| WC = DW | 0.455 | 0.347 | +0.108 | [0.08, 0.14] | [0.10, 0.12] |
| WC = ESRI | 0.476 | 0.348 | +0.128 | [0.10, 0.15] | [0.12, 0.13] |

条件独立期望值用样本自身的 P(A=c \| Y=j) 代入估计——这会**偏向拟合独立**，
所以 excess 是保守下界（写作时点明）。DW=ESRI 的 excess（+0.24）约为 WC 配对
（+0.11 / +0.13）的两倍：任意两个真实产品都会共享一点类别边缘以外的信号，
但 DW/ESRI 的耦合明显更强。

**S2b — 两个产品「相对专家的错误」之间的依赖（集成多样性统计）**

| 错误对 | 错误率 A / B | 共错率 obs / 独立期望 | Yule's Q（+CI） | κ(错误指示) | 都错时同一错标 |
|---|---|---|---|---|---|
| **DW, ESRI** | 0.57 / 0.58 | 0.52 / 0.33（×1.57） | **0.969** [0.95, 0.98] | 0.77 | 0.96 |
| WC, DW | 0.54 / 0.57 | 0.30 / 0.31（×0.98） | **−0.045** [−0.19, 0.10] | −0.02 | 0.87 |
| WC, ESRI | 0.54 / 0.58 | 0.32 / 0.32（×1.02） | **0.054** [−0.09, 0.20] | 0.03 | 0.86 |

**这是本节最锋利的结果**：DW 和 ESRI 的错误几乎共动（Q ≈ 0.97，都错时 96%
选同一个错类）；而 WC 的错误与 DW、与 ESRI 都**统计独立**（Q ≈ 0，CI 跨 0）。
即"WC 独立地犯错，DW 和 ESRI 一起犯错"。（"都错时同一错标"这个指标对 WC 配对
也有 0.86，因为三类里两个产品都错时常落在同一主混淆方向 built↔non-built 上——
所以主要靠 Q / κ 区分，不靠这一列。）

**S2c — 产品间「一致」能不能预测专家标签？（仲裁假设的直接检验）**

| 条件 | n | P(专家 = 该标签) | vs「只问一个产品」基线 |
|---|---|---|---|
| 基线 P(专家 = DW) | 750 | 0.431 | — |
| 基线 P(专家 = ESRI) | 750 | 0.416 | — |
| **DW = ESRI** → P(专家 = 该标签) | 652 | **0.423** [0.39, 0.46] | ≈ 持平（+0.15 只是相对 0.27 的类别先验） |
| DW = ESRI = WC → P(专家 = 该标签) | 306 | 0.395 [0.34, 0.45] | 略低 |
| WC = DW → P(专家 = 该标签) | 341 | 0.422 [0.37, 0.48] | ≈ 持平 |

在 DW=ESRI 的 652 个点（占 87%）上，专家跟从这个「共识」的概率是 0.42——
和「只问 DW」的 0.43 **基本一样**。三产品全一致也只有 0.40。
**产品间达成一致，几乎不携带关于真值的额外信息**——这正是仲裁失效的定量表述。

**S3a — 排混淆项①：类别体系**

| 方案 | DW=ESRI | Human=DW | gap |
|---|---|---|---|
| 3 类（建成/非建成/水体） | 0.869 | 0.431 | 0.44 |
| 2 类（建成 vs 其余） | 0.905 | 0.545 | 0.36 |
| 建成 vs 植被（去掉水体点） | 0.915 | 0.440 | 0.48 |

把类别并粗**不会缩小 gap**（DW=ESRI 稳定在 0.87–0.92，与专家的差距稳定在
0.36–0.48）——不是 3 类映射造成的假象。

**S3b — 排混淆项②：共享输入影像 / WorldCover 作阴性对照**
WC 同样以 Sentinel-2 为输入，但与 DW/ESRI 的 excess（+0.11 / +0.13）约为
DW=ESRI（+0.24）的一半，且错误层面 Q ≈ 0。DW/ESRI 相对 WC 多共享的是
**深度语义分割方法族 + S2 合成影像上的点标注语料谱系**（WC 是 boosted-tree +
规则流程）。"共享 S2 输入"本身产生不了这个耦合。

**S3c — 排混淆项③：同影像同一天？**
DW（准瞬时）与 ESRI（年度合成）时相处理不同，但 DW=ESRI 一致率在**所有场景类
都高**（0.81–0.96），在「建成区边缘」「Mixed/complex」等最难的转换带反而最高
（0.95+）。若耦合来自「看的是同一天同一景」，应集中在稳定类、在转换带瓦解——
观测到的是近均匀，方向相反。

**叙事句（4.1b）**：DW 与 ESRI 在城市边界上不满足「给定真值条件独立」——
错误共动 Yule's Q ≈ 0.97，产品间一致相对条件独立期望多出 0.24（保守下界），
且这一耦合对专家判断几乎零增益（仲裁 PPV 0.42 ≈ 单产品 0.43）。类别体系、
共享 S2 输入、时相三个廉价解释均已排除。"为何耦合"（共享方法族/标注谱系）
留待 Discussion 作假设，不作本文结论。

### 4.2 按置信度分层 (T2) — `T2_confidence_stratified.csv`

Human ≠ WC 的比例：

| 层 | 6 城合并 | Wuhan | Hefei | Nanchang | Nanjing | Changsha | Hangzhou |
|---|---|---|---|---|---|---|---|
| 全部手标 | 0.53 (n=900) | 0.59 | 0.49 | 0.45 | 0.68 | 0.49 | 0.47 |
| high+med | 0.55 (n=806) | 0.58 | 0.49 | 0.45 | 0.68 | 0.57 | 0.53 |
| **high only** | **0.67 (n=436)** | **0.88** | 0.53 | **1.00** | 0.62 | 0.58 | 0.58 |

**叙事句**：专家最有把握的地方，WC 不是更对而是更错（合并 0.53 → 0.67；
Nanchang 42 个 high 点 100% 与 WC 分歧）。**必须同时写明抽样偏**：见 §7.1。

### 4.3 WC 边界误差的方向 (T3) — `T3a_WC_to_human_confusion.csv` / `T3b_WC_error_direction.csv`

6 城合并，边界候选点。WC→Human 混淆（计数）：

| WC＼专家 | 建成 | 非建成 | 水体 |
|---|---|---|---|
| WC=建成 (n=263) | 70 | **181** | 12 |
| WC=非建成 (n=510) | **167** | 284 | 59 |
| WC=水体 (n=161) | 10 | 46 | 71 |

导出：P(专家=非建成 | WC=建成) = **0.69**；P(专家=建成 | WC=非建成) = **0.33**；
净方向偏差 net_built_over_call = 0.017（很小；2026-09-06 南昌标签更正后由 0.016
微调，四舍五入前的其余数字不受影响）。

**叙事句**：在边界候选点上，一旦 WC 判「建成」，约 7 成其实是植被；是双向混淆，
但「WC 一旦承诺建成就大概率错」这个条件不对称是可用的。

### 4.4 物理场景 × 分歧 (T4) — `T4_scene_disagreement.csv`

⚠️ **2026-09-06 场景代码映射已更正**（见决策日志 + §6.3）：原始短代码 SB/BS 在早期
分组脚本里搞反了，UB/BR/WR/PF 也被错误归类。用标注者提供的原始代码表核对后重新
分组、重跑，下表已是更正后的版本，**旧版本（含"Soft boundary / shadow"26点/65.4%
这一行）不再有效，不要再引用**。

5 城合并，按分歧率排序（n ≥ 15 的组）：

| 场景组 | n | 分歧率 | 95% CI |
|---|---|---|---|
| Built-up edge / urban fabric | 155 | 69.0% | [61,76] |
| **Urban green space**（新类别） | 42 | 69.0% | [54,81] |
| Road / road edge | 106 | 63.2% | [54,72] |
| Water / water edge | 152 | 58.6% | [51,66] |
| Vegetation / forest edge | 97 | 55.7% | [46,65] |
| Paddy / cropland / field | 17 | 41.2% | [22,64] |
| Rural settlement / surfaces | 39 | 41.0% | [27,57] |
| Bare land / construction | 63 | 39.7% | [29,52] |

（Greenhouse / polytunnel 9 点 11%，n 太小；"Soft boundary / shadow" 更正后仅 3 点、
0% 分歧，跌出主表，**不再是论文可用的类别**；Nanchang 的 26 个 "Unlabeled" 点已在
`diagnostic_analysis.py` 的 `table_scene()` 里显式剔除，不进任何统计——核实过这
26 点 Human=WC 100% 一致，缺失非随机（标注协议下只在分歧点写场景说明），不影响
排序有效性，见 §7.2）

**叙事更新**：headline 结论不变——Built-up edge 仍是分歧率最高的场景类别，数字
基本没动（68.8%→69.0%）。但新增了一个真实发现：**Urban green space（城市绿地，
42点）分歧率与 Built-up edge 并列第一（69.0%）**——呼应 §6.4 检索到的文献
（"urban green areas classified as built areas"），建议在 Discussion 里正面
论证这一点，是本文场景诊断的一个新增亮点，不只是数据修正。

### 4.5 PE 作废的证据（背景，写作一般不展开）

- 泄漏诊断 `docs/results_snapshot/leak_diagnostic/leak_diagnostic.csv`：
  5 城 × K∈{5,10,20,30} 全部 `leak_fraction = 1.0`。
- 空间分块 + 显式排除测试点重跑：`spatial_block_results.csv` 里 `n_expanded` 全为 **0**。
- `spatial_block_summary.csv`：Baseline OA 0.9138 / BE 275.7；BAMS150 0.9137 / BE 275.3；
  「PE_Boundary」OA 0.9242 / BE 226.9 —— 但 n_expanded=0，所以这 +1pp 全是
  RF→MLP + 锚点，不是近邻传播。
- 消融（Wuhan, 分块种子0）：RF baseline .9053/BE291 → MLP(仅train) .9145/BE254
  → MLP+BAMS锚点 .9171/BE250 → 「PE_Boundary」.9197/BE234（n_expanded=0）。

### 4.6 LOCO 校正算子（支撑章节）— `docs/results_snapshot/loco_correction_operator/`

**Phase 1 — 留一城迁移**（算子 = RF，27 特征，其他城约 750 手标点训练，置信度加权；
`phase1_corrector_transfer.csv`，用 `feature_set=all27`）：

| 留出城 | WC↔Human | 算子↔Human | 高置信触发数(τ=0.85) | 触发精度 vs Human |
|---|---|---|---|---|
| Wuhan | 0.41 | 0.72 | 14 | 0.71 |
| Hefei | 0.51 | 0.66 | 28 | 0.68 |
| Nanchang | 0.55 | 0.68 | 11 | 0.45 |
| **Nanjing** | **0.32** | **0.75** | 12 | **1.00** |
| Changsha | 0.51 | 0.74 | 12 | 0.92 |
| Hangzhou | 0.53 | 0.71 | 3 | 1.00 |

`transfer15` 特征子集整体略差于 `all27`（原始波段即便有跨城偏移仍有用）。
（2026-09-07 重跑：数字相对 Sep-3 冻结集有微移，来源是南昌 NA02 / Original_ID
4343 的 inter-annotator 复核改判 Human 1→2 经跨城算子传导，非 Hangzhou DW/ESRI
的改动——Phase 1 一直是 6 城口径。）

**Phase 2b — gate 敏感性**（6 城均值，基线 Human_OA ≈ 0.40 / WC_OA ≈ 0.93；
`phase2b_sweep_summary.csv`）：

| τ | margin 分位 q | 改动点数 | Human_OA | WC_OA |
|---|---|---|---|---|
| 0.70 | 0.20 | 465 | **0.509** | 0.907 |
| 0.70 | 0.10 | 254 | 0.457 | 0.922 |
| 0.85 | 0.10 | 64 | 0.414 | 0.927 |
| 0.90 | 0.10 | 16 | 0.403 | 0.929 |
| baseline | — | 0 | 0.400 | 0.929 |

→ 要 +10pt Human_OA 得在 τ=0.70 改约 465 点并损失 2pt WC_OA；安全阈值下效果落进噪声。
**下游 RF 分类器不是合适的评测载体**——几十个边界标签撼动不了 1 万点训的 RF。

**Phase 3 — 直接评「修正后的标签图」，DW/ESRI 当裁判**（**6 城** × 150 点均值，
2026-09-07 Hangzhou 加入后重跑；`phase3_referee_summary.csv`）：

| 方法 | OA vs Human | OA vs 三方共识子集 | 改动点中→Human | →DW/ESRI | 改动点数 |
|---|---|---|---|---|---|
| WC 原始 | 0.47 | 0.44 | — | — | 0 |
| **LOCO 修正 (τ=0.85)** | **0.52** | 0.46 | **0.79** | 0.41 | 13 |
| 随机翻转对照 | 0.46 | 0.44 | 0.30 | 0.18 | 13 |
| 本城算子（天花板，乐观） | 0.80 | 0.76 | 1.00 | 0.42 | 49 |

（旧 5 城值：WC 0.46 / LOCO 0.49（→Human 0.72）/ 天花板 0.80。Hangzhou 单城只改
3/150 点，3 点全部朝专家、也全部朝 DW=ESRI，OA 0.53→0.55——与 Phase 1 fires=3、
`Fig_diag_map_Hangzhou` 稀疏改动一致。）

**叙事句**：LOCO 每城改约 13 点，其中 79% 朝专家方向（随机对照 30%）——信号真实、
非噪声；但对整图质量提升有限（+4–5pt vs Human，对三方共识子集几乎不动
0.44→0.46），只追回本城算子（天花板）增益的约 1/7（OA 口径：+7 点 / 天花板
+48 点）。**南京是诊断意义上的极值点**：六城中 WC-Human 一致率最低（0.32），
同时也是跨城算子绝对提升最大、触发精度最高的城市（算子↔Human 0.75，12 次高置信
触发全部命中专家）——这是一个诊断事实（参考冲突最大的地方留给迁移算子的提升
空间也最大），不是"failure case 被反转"这种 PE 时代的方法成败叙事，写作时不要
带出旧逻辑的措辞。

### 4.7 边界修正前/后可视化地图（全 6 城，主文 2 + 补充 4）— `scripts/make_correction_maps.py`

三联图（每城一张，`Fig_diag_map_{City}.png`，**全 6 城都已生成**）：
A) WC 原始类别地图（15000 点）；B) LOCO 修正后地图，圈出算子改动的点；
C) 150 个手标边界点按 human=WC / human≠WC 着色叠在淡化背景上。用的正是 §4.6
同一个 τ=0.85 / 最低 10% margin 的 gate，和 T5 的算子完全一致（不是另起一套）。

| 城市 | 算子改动点数 (/15000) | 150 点中 human≠WC | 用途 |
|---|---|---|---|
| **Nanjing** | 80 | 102 | 主文——六城中 WC-Human 一致率最低(0.32)，算子提升也最大 |
| **Changsha** | 138 | 74 | 主文——中等分歧城市，算子触发精度高(0.92) |
| Wuhan | 84 | 88 | 补充材料 |
| Hefei | 148 | 74 | 补充材料 |
| Nanchang | 126 | 68 | 补充材料 |
| Hangzhou | 21 | 70 | 补充材料——算子触发数最少，与 Phase1 fires=3 一致 |

（2026-09-07 重跑值；旧值 Nanjing 85 / Changsha 142 / Wuhan 83 / Hefei 143 /
Hangzhou 18，Nanchang human≠WC 67→68 是 NA02 改判所致。差异都在个位数，不影响
"改动点沿建成边缘/水岸聚集" 的定性结论。）

⚠️ **措辞提醒**：南京在这里只是「六城中 WC-Human 一致率最低 + 算子迁移增益最大」
两个诊断事实的叠加，**不要**写成「failure case 被反转」——那是 PE 时代
（方法有效/失效叙事）的残留说法，诊断优先框架下没有"失败案例"这个概念。

**观察**：六城的算子改动点和人-WC 分歧点都清楚地沿着建成区边缘、江/水体沿岸
聚集（而不是散布在整张图上）——直观佐证了「BAMS 边界候选采样确实抓在了空间
上有意义的边界地带」，可以放进 Results 4.6 作为定性支撑，并回应 §7 的采样
局限（虽然不能证明"比随机采样分歧率更高"，但空间聚集模式本身是有信息量的）。

---

## 5. 文件位置总表

### 5.1 结果快照（复跑不覆盖，写作引用这里）
`docs/results_snapshot/`
- `SNAPSHOT_INFO.txt` — 快照时间与来源
- `diagnostic_analysis/` — T1–T5 CSV + SUMMARY.txt
- `loco_correction_operator/` — phase1/2/2b/3 全部 CSV + run.log + phase1 txt
- `leak_diagnostic/` — leak_diagnostic + spatial_block_results/summary
- `figures/` — Fig_diag_{fourway,confidence,scene,error_direction}.png +
  Fig_diag_map_{City}.png（全 6 城：Wuhan/Hefei/Nanchang/Nanjing/Changsha/Hangzhou）

### 5.2 实时输出（脚本每次重写）
- `data/analysis_outputs/{diagnostic_analysis,independence_diagnostic,loco_correction_operator}/`
- `figures/Fig_diag_*.png`

### 5.3 原始数据资产
| 内容 | 路径 | 说明 |
|---|---|---|
| 手标边界样本（6 城） | `data/cities/{City}/04_manual/{City}_BAMS150_Manual.csv` | PointID,Original_ID,Human_Class(1/2/3),Scene,Confidence(h/m/l),Notes |
| 15000 点 + WC + 特征 | `data/cities/{City}/01_original/{City}_WC_Samples_15000.csv` | B2..B12,NDVI/NDBI/MNDWI,Class(WC),Original_ID,lat,lon |
| 3×3 邻域特征 | `data/cities/{City}/06_3x3/{City}_3x3_Features.csv` | 各波段/指数 _mean,_stdDev |
| margin 分数 | `data/cities/{City}/02_margin/{City}_MarginScores.csv` | margin,prob_1/2/3 |
| 第二/三参考 DW+ESRI | `data/shared_reference/cross_city_second_ref_dw_esri/all_cities_BAMS150_second_ref.csv` | 5 城×150，含 DW_C3/ESRI_C3；**无杭州** |
| 场景分组表 | `data/shared_reference/cross_city_scene_localization/bams150_scene_grouped.csv` | 750 行，Scene_group,Disagree,Confidence |
| 南京 root-cause | `data/nanjing_rootcause/nanjing_baseline_autopsy/VERDICT.md`,`data/nanjing_rootcause/nanjing_rootcause_v2/ROOTCAUSE_STATUS.md` | C2 类混淆预算，Discussion 用 |
| 南京 Round2 +50 点 | `data/cities/Nanjing/04_manual/Nanjing_Round2_C12Boundary50.csv` | 可选并入算子训练 |
| 泄漏诊断脚本 | `../gee-project2/scripts/spatial_block_validation.py` | 结果在 `../gee-project2/data/spatial_block_validation/` |
| 给导师旧报告 | `docs/supervisor_progress_report_latest.md` | 仍是 PE 叙事，需更正 |

---

## 6. 待办清单（按优先级；每项含 做什么/输入/产出）

### 6.1 ~~1–2 城边界修正前/后可视化地图~~ ✅ 已完成 2026-09-03
`scripts/make_correction_maps.py` → `data/analysis_outputs/correction_maps/Fig_diag_map_{City}.png`，全 6 城（见 §4.7）。

### 6.2 ~~复核 Changsha 的 WC↔Human 分歧率~~ ✅ 已复核 2026-09-05
**结论：`diagnostic_analysis.py` 口径没问题，读的就是每城最终修正版
`04_manual/{City}_BAMS150_Manual.csv`；数字权威，无需改代码。**

- 旧表 `cross_city_wc_reference_bias/bams_human_vs_wc_by_city.csv`
  生成于 2026-08-07 21:46，是**标注质检/修正完成之前**的快照，之后未重新生成，
  是**过期文件**——已于 2026-09-06 清理时随整个 `cross_city_wc_reference_bias/` 删除
  （见 `docs/CLEANUP_LOG.md`）。此处仅记录历史结论，勿再找该文件。
- Changsha：52.7%（旧）→ 49.3%（现）。对比
  `Changsha_BAMS150_Manual_before_100.csv`（08-09 16:39）与最终版
  （08-09 17:21）：9/150 点被改判——4 个道路/桥梁场景（路面/城市路面/bridge/
  road）从"非建成"改判"建成"，3 个水体点改判确认，2 个其他。
- 顺带查出 **Hefei 差异更大**：旧表记 81.3%，现为 49.3%。对比 `_bk` 备份
  （08-08 09:50，无 Confidence 列、是质检前粗标稿）与最终版
  （08-08 12:02，加了 Confidence 分级）：58/150 点被改判。旧表用的是粗标阶段
  的数字，不能用。
- Wuhan/Nanjing/Nanchang 无此问题：最终版与旧表数字几乎/完全一致
  （Nanjing 0.68、Nanchang 0.4467 完全相等），说明这三城旧表生成时已定稿。

### 6.3 ~~清理场景表~~ ✅ 已完成 2026-09-06（范围比原计划大——发现并更正了映射错误）

- 用标注者提供的原始场景代码表（18码）核对，发现 SB/BS 搞反、UB/BR/WR/PF 归类错误，
  涉及 76 个点，详见决策日志 + §4.4。
- ~~Nanchang 26 个 "Unlabeled" 点已在 `diagnostic_analysis.py` 里显式剔除，核实过是
  系统性缺失（只在分歧点写场景）。~~ **更正 2026-09-08**：这 26 行不是"只在分歧点写
  场景"的系统性缺失，而是标注时的录入失误——`Human_Class` 被直接拷成了 WC 值、
  `Scene` 漏填（所以 26 行全部 Human==WC、Scene 空，只在南昌出现，其余 5 城 0 个）。
- **2026-09-08 已全部重标**：26 个点在 GEE 影像上双人复核，补齐 `Human_Class` + `Scene`，
  13/26 相对 WC 占位值改判。脚本 `scripts/gee_nanchang_scene26_review.js`。
  已改：`Nanchang_BAMS150_Manual.csv`（26 行，Notes 内联标注，参照 PointID 97 体例）、
  `bams150_scene_grouped.csv`（26 行并入，T4 `n 724→750`）。
  已重跑：`diagnostic_analysis.py` / `independence_diagnostic.py` /
  `loco_correction_operator.py` / `make_correction_maps.py`。
  影响：南昌 H≠WC `0.453→0.540`；Pooled H=WC `0.47→0.46`；T1/T2/T4/T5/Phase3/S2/S3
  及 before/after maps 表均刷新；LOCO 全 6 城 T5 数字变（南昌在其余 5 城训练折内，
  确定性传播，非随机）。manuscript.md 对应处已改（§3.2d、§4.1、§4.4、§4.5、§4.6、§4.7、
  附录表）。
- ⚠️ 待办：`docs/results_snapshot/` 是冻结快照，尚未随本次更正重新生成；如要引用需刷新。
- T4 表、`scene_group_by_city.csv`、`figures/Fig_diag_scene.png` 均已同步刷新。
- ⚠️ **之前发给导师的简报 artifact 里的 T4 数字是旧版（含已作废的"Soft boundary /
  shadow"）**，如果还要用那份简报，需要重新发布同步这次的更正。

### 6.4 【中】Related Work 调研定稿 —— 2026-09-05 已跑一轮系统检索，新颖性缺口成立但要扩到 5-6 篇引用

**结论：没有检索到直接重叠的论文（边界像元级 + 主动采样极小人工预算 + 显式检验
「多产品能否互相仲裁」）。但必须正面引用以下文献说明「分歧本身非新发现，
新的是边界级 + 仲裁假设检验」，不能对已有比较类文献只字不提。**

**桶 A：WC/DW/ESRI 直接比较类（必引，逐条写清差异）**
- **Venter et al. 2022, *Remote Sensing* 14(16):4101**（同期刊！）"Global 10 m LULC
  Datasets: A Comparison of DW, WC and Esri Land Cover"——全球尺度地面真值点精度
  比较（ESRI 75%>DW 72%>WC 65%），报告系统性类别偏差（WC 偏高估草地/ESRI
  偏高估灌木/DW 偏高估雪冰）。**无**边界像元聚焦、**无**人工标注边界候选点、
  **无**仲裁假设检验——纯描述性精度报告。
- **Xu et al. 2024, *Remote Sensing of Environment*** "Comparative validation of
  recent 10 m-resolution global land cover maps"——同样比较 WC/DW/ESRI，全球独立
  验证集，总体精度 73–83%。同样是全局尺度描述性验证，非边界导向。
- **IJDE 2024 (Tandfonline, doi 10.1080/17538947.2023.2301673)** "Evaluation of six
  global high-resolution land cover products over China"——中国全域比较 6 产品
  （含 WC10/ESRI10，不含 DW），发现「一致性低于精度」现象，但全国尺度描述性
  统计，无边界诊断、无人工锚点。**可用于支撑「中国范围内已知产品间低一致性，
  但未细究边界机制」这个引子。**
- **Nature Communications 2024** "Large disagreements in estimates of urban land
  across scales and their implications"——讲的是不同产品对**城市面积**这个宏观
  统计量的分歧，不是像素级边界分类分歧，维度不同，简单一句带过即可。
- **Consensus land cover 一脉**（EarthEnv 等）——默认假设「多产品加权合并可逼近
  真值」，与本文「证明合并/仲裁在边界像元上不成立」方向相反，可作 Introduction
  的靶子（"过去默认可仲裁/合并，我们检验后发现在边界像元上不成立"）。

**桶 B：弱监督/原型类方法（原有两篇，确认关系）**
- *A cross-spatiotemporal weakly supervised framework*（ScienceDirect 2025）——
  解决 HR 影像与公开 LC 产品的时空分辨率错配，非边界仲裁问题，角度不同。
- ⚠️ **arXiv 2406.00891 / JPRS 2025（Tong, Dong, Zhu）"Global High Categorical
  Resolution Land Cover Mapping via Weak Supervision"，方法名 PRE
  (Prototype-based pseudo-label Rectification and Expansion)**——**几乎可以确定
  是本项目已作废的 Prototype Expansion (PE) 方法的直接源头**（原型/特征中心
  做伪标签修正扩张，机制对应；PRE 原论文在 10 个跨大洲城市上测，PE 在本项目
  6 城内部测）。**写作定案（2026-09-05，见决策日志 + §3.3 Methods 大纲）：
  正面写出这段弯路，不回避**——点名引用 PRE，讲清楚「同域内原型/近邻传播在
  本项目场景（同城内部边界像元、空间自相关强）下会结构性泄漏」，把 LOCO 的
  跨城留一设计写成排除法排出来的方法论依据，而不是隐瞒的失败史。具体段落
  写法见 §3.3。

**差异化论点（写进 Introduction 结尾/Related Work 收尾段）**：
以上文献都证明「产品间存在分歧」，但没有一篇做到本文这个组合——
① 专门聚焦建成↔植被边界像元而非全图；② margin + 局部光谱异质性主动采样，
把 900 个标注点精确砸在难点上；③ 把「加第二/三个产品能否仲裁」这个默认假设
形式化为**条件独立性的可检验违背**并给出定量证据（DW/ESRI 错误 Yule's Q ≈ 0.97、
相对条件独立期望多出 0.24、仲裁 PPV 0.42 ≈ 单产品 0.43），且排除了类别体系/
共享输入/时相三个廉价解释——"为何耦合"作为 Discussion 假设，不写成结论。

### 6.5 【低/可选】LOCO 算子加强
- 加 3×3 WC 邻域上下文特征；并入南京 Round2 +50 点；逐城校准 τ；试 GBDT/集成。
- 预期把 Phase3 OA-vs-Human 从 0.49 提到 ~0.55，**到不了 0.80**。若时间紧可跳过。

### 6.6 【独立】更正给导师的 2026-08-08 报告（PE 归因）

### 6.8 【中】Inter-annotator 盲标复核 —— 2026-09-06 六城全部完成 ✅

**背景**：第一次尝试（3 位同事对照原始标注表打对错）经技术核查确认不是独立盲标——
Scene/Confidence 字段与原始文件 150/150 完全一致，只有 Human_Class 改了 2-3 处，
判定是"审核已有答案"而非"从零独立标注"，**该批数据已弃用，不进论文**。

**重新发起的协议**（已生成，见下）：
- 每城分层抽样（按原 Confidence 比例）20 点，6 城共 **120 点**
- 三位同事（A/B/C）**每人独立标全部 120 点**（不分工，kappa 需要同一批点被多人标）
- 发给同事的模板**只含** CheckID/Original_ID/lat/lon，不含任何参考答案（Human_Class/
  Confidence/Scene/Notes 全部清空），协议说明见 `使用说明.md`
- 待同事填完，放回 `returned/annotator_{A,B,C}/{City}_Recheck20_BLANK.csv`，跑
  `scripts/score_interannotator_recheck.py` 算 Cohen's kappa（每人 vs 参考）+
  Fleiss' kappa（三人互相之间，不依赖参考标签）

**文件位置**：
- 发给同事：`data/interannotator_recheck/to_send/annotator_{A,B,C}/` + `使用说明.md`
- 参考答案（内部使用，不发给同事）：`data/interannotator_recheck/answer_key/recheck_answer_key.csv`
- 同事填完后放这里：`data/interannotator_recheck/returned/annotator_{A,B,C}/`
- 评分脚本：`scripts/score_interannotator_recheck.py` → 输出到 `data/interannotator_recheck/results/`

**用途**：结果写进 Methods 标注可信度段落 + 补强 §7.4（单一标注者局限）。若这次同事
仍未能认真独立完成，退回方案：论文老实写"未做正式 inter-annotator 统计检验"，
用标注者本地领域经验 + 引用 Olofsson et al. (2014) 等参考数据采集规范背书。

**结果 —— 2026-09-06 六城全部完成 ✅（120/120 点全部填齐，无缺失）**

6 城 × 20 点分层抽样（120点），A/B/C 三人独立盲标，全部完成。（此前 B/C 有 4 点
一度显示缺失，实为返回文件未同步，补齐后确认无遗漏。）

| 城市 | A kappa | B kappa | C kappa | Fleiss' κ |
|---|---|---|---|---|
| 武汉 | 1.000 | 0.903 | 1.000 | 0.934 |
| 合肥 | 1.000 | 0.811 | 1.000 | 0.873 |
| 南昌 | 0.895 | 1.000 | 0.895 | 0.857 |
| 南京 | 1.000 | 0.897 | 0.908 | 0.871 |
| 长沙 | 0.822 | 0.913 | 0.913 | 0.939 |
| 杭州 | 1.000 | 0.897 | 0.897 | 0.860 |

**总体（120点，全部完整）**：A kappa=0.952（一致率97.5%）、B kappa=0.905（95.0%）、
C kappa=0.937（96.7%）。**Fleiss' kappa = 0.892**——全部落在 Landis & Koch
"几乎完全一致"区间（>0.81）。

**分歧模式**：**13处分歧 / 360份判断（3.6%）**（注：早前统计误将 4 个当时未同步、
显示为空的格子按 pandas `!=` 比较误判为"分歧"计入 17，补齐同步后更正为 13，是
真实分歧数）。约9处集中在水体边缘/水陆过渡带（长沙CH20三人全部一致地跟参考不同、
南昌NA08/NA13、南京NA11/NA12、杭州HA09/HA13、合肥HE14、武汉WU15），跨六城稳定
重复——建议 Discussion 专辟一段，用具体例子论证"边界本身存在专家间也无法消除的
内在不确定性"。另两处例外：长沙CH12（路面）、合肥HE03（建成区内绿地，呼应
Related Work 检索到的"城市绿地易与建成混淆"文献）。

**过程中发现并更正一处原始标签**：南昌 NA02（Original_ID 4343）三人独立全部判"非建成/
水体边缘"，经核实原参考标签（建成，场景码"be"）确系误标，已更正为 Human_Class=2、
Scene="水体边缘"，同步改了 `data/cities/Nanchang/04_manual/Nanchang_BAMS150_Manual.csv` 和
answer_key。该点原 WC_Class=1 与旧 Human_Class=1 一致，更正后变为 Human≠WC。

✅ **2026-09-06 已重跑核对**：`diagnostic_analysis.py` + `independence_diagnostic.py`
重跑，快照已同步更新（见 `docs/results_snapshot/SNAPSHOT_INFO.txt`）。T1/T2/T3a/T3b
CSV 有变化，但**变化全部在小数点后第 3 位，§4 引用的所有 2 位小数数字四舍五入后
不变**，唯一改写的是 T3 的 net_built_over_call（0.016→0.017，仍是"很小"）。T4
（读的是独立冻结的场景分组表，未联动，随 §6.3 一起处理）和 T5/LOCO（该脚本不重训）
不受影响；independence_diagnostic 六张表全部逐字节不变。**数据侧闭环，无需再跑。**

**文件**：完整数据 `data/interannotator_recheck/results/{city_summary_final,
merged_scored_final,disagreements_final}.csv`。

**Methods 写作草稿**（标注可信度段落，可直接用）：
> To assess the reliability of the single-annotator reference labels, three
> independent colleagues (all with remote-sensing land-cover interpretation
> experience) each blindly re-labelled a stratified subsample of 120 boundary
> points (20 per city, proportional to the original confidence-tier distribution),
> without access to the original labels, WC/DW/ESRI outputs, or each other's
> judgements. Agreement with the reference labels was substantial to almost
> perfect for all three annotators (Cohen's κ = 0.95, 0.91, 0.94; Landis and Koch,
> 1977), and inter-annotator agreement among the three was likewise almost
> perfect (Fleiss' κ = 0.89). The 13 disagreements (3.6% of 360 judgements) were
> not randomly distributed: roughly 70% concentrated at water–land transition
> zones, recurring across all six cities — indicating that even independent
> domain experts retain some irreducible disagreement at these specific boundary
> types, consistent with the diagnostic finding that boundary pixels carry
> genuine interpretive ambiguity rather than simple annotation noise.

### 6.9 【收尾】全文成稿 + 过 RS checklist
新颖性 / 空间独立验证 / 外部参考 / 统计严谨性（CI、对照）/ 可视化。

---

## 7. 已知问题 / 审稿人会问的点（提前想好答复）

### 7.1 置信度分层是有偏样本
high 子集里 WC 分歧率更高——因为标注者很可能正是在「明显是建筑却被标成植被」
这种清楚的 WC 错误上给了 high。**文中必须先承认这个偏**，再论证：即便如此，
（a）它反驳了「人标在模糊像元上只是噪声」的说法；（b）Nanchang 100%、Wuhan 88%
的量级不是抽样偏能完全解释的。

### 7.2 场景分组不是正式分层抽样
关键词式分组，仅供诊断定位（见 `cross_city_scene_localization/README.md`）。
Wuhan/Changsha/Nanchang 用短代码、Hefei/Nanjing 用中文自由文本，大棚/温室等标签
几乎只出现在后两城——写作时说明，不要跨城比大棚。

### 7.3 一致率 ≠ 全图精度
最重要的一条，见 §4.0。BAMS 采样富集难点，所有数字是「边界候选点上」的。

### 7.4 没有独立的「真值」
Human 是唯一相对可信锚，但也只是一个专家；DW/ESRI 互相相关不能当独立真值
（§4.1b 已定量：错误 Yule's Q ≈ 0.97）。论文的定位是「量化分歧 + 证明多产品
不仲裁」，不是「宣布谁对」。
- **审稿人反打**：n=1 标注者时，可以说「DW/ESRI 一致是因为它俩都对，是你的
  标注者『建成』阈值特殊」。防守：§4.1b 的独立性结果只依赖「三个产品之间」的
  错误结构（WC 与 DW/ESRI 的 Q ≈ 0 vs DW–ESRI 的 Q ≈ 0.97），这个对比**不需要
  假定专家是真值**——一个用 boosted-tree、两个用深度分割，后两者错误共动而前者
  独立，本身就是产品侧的证据。但「仲裁 PPV」「置信度分层」等确实依赖专家锚，
  这部分的强度上限由 §6.8 的 inter-annotator κ 决定，两件事耦合，见 §6.8 退路。
- **S2a 期望值的口径**：条件独立期望用样本自身的条件类分布代入，偏向拟合独立，
  故 excess 是保守下界——写作时明写这一点，把它当成对我们有利的方向。
- **参考与选点共享 S2 信息基（2026-09-07 补，标注主判据定为 S2 后必须回应）**：
  BAMS 用 S2 光谱 RF 的 margin + S2 局部异质性挑点，参考标签又主要从 2021 S2
  RGB 判读——审稿人会问「测的是产品误差，还是 S2 本身难读」。防守 4 点，写进
  §3.1 "Sampling-induced conditioning" 段 + §5：
  ① 人整合空间格局/纹理/上下文/假彩 NIR + 不清楚时切 Google 底图，不是又一个
     光谱像素分类器；② inter-annotator κ = 0.89：三个独立专家从同一影像收敛，
     标签可复现非噪声；③ 分歧集中在**高置信**标签（T2）+ **可解释场景**（T4），
     模糊像元噪声不会是这个分布；④ 最锋利的结果 DW/ESRI 错误 Q ≈ 0.97 是
     **产品侧**的，根本不需要参考标签。
  可选加分（需用户，视觉活）：挑 ~30 个高置信分歧点在亚米 Google 影像上复核，
  站得住就写一句 robustness check。

### 7.5 LOCO 影响适度
诚实写：算子迁移信号真实（72% 朝专家方向），但只追回本地标注天花板的 ~1/4，
下游分类器层面提升在安全阈值下落进噪声。这是支撑章节，不是主卖点。

### 7.6 无版本控制
`gee-project` 不是 git 仓库。结果靠 `docs/results_snapshot/` 冻结。重大改动前手动备份。

### 7.7 地理泛化局限（2026-09-06 补，讨论已久但一直没落笔）
六城全部在长江中下游/流域周边，单一国家、单一气候带（亚热带季风）、单一传感器
（Sentinel-2）、同一批中国城市化建成形态。全文任何结论都不能写成"全球产品普遍
如何如何"，必须老实框定为"在长江中下游六城的案例中"。
- **不是致命伤**：RS 大量发单一区域案例研究，只要不超范围声称、并在 Limitations
  主动承认，通常只招致"讨论可推广性"这类审稿意见，不至于因此被拒。
- **可用的内部稳健性论据**：六城内部并不同质——南京(Human=WC 0.32，全场最低)、
  杭州(0.53，全场最高)、南昌高置信点 100% 分歧——现象在不同城市表现不同但方向
  一致，说明不是单一城市的偶然 artifact。
- **可区分"数值局限"与"机制可能泛化"**：DW/ESRI 耦合的机制解释（共享深度分割
  方法族 + S2 点标注语料谱系，见 §4.1b/S3b）是这两个模型全球统一训练带来的架构
  性质，不是长江流域地理特有的，具体数值（0.87、Yule's Q≈0.97 等）不会照搬到
  其他地区，但"耦合机制"这个定性结论大概率能推广，值得在 Discussion 里明说
  这层区分。
- **写作要求**：Study Area 用"区域实际监测需求"框定选点（§1.2 决策日志
  2026-09-05），不写全球普适性断言；Discussion/Limitations 单独一段承认范围
  局限，建议未来工作在不同气候带/城市形态复现同一套 BAMS+四方诊断协议。

---

## 8. 备注

§1/§3/§4 的诊断优先框架定于 2026-09-03，基于对 `gee-project` 数据的实际读取与
本地跑通（脚本见 §2，输出见 §5）。此前所有给导师的进展报告都还停留在 PE 叙事，
写作以本文件为准，勿混用。
