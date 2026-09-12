# 代码/文件清理记录

> 项目非 git 仓库，删除不可逆。本文件记录被删内容与去向，供事后追溯。
> 权威说明见 `docs/Paper_Writing_Plan.md`（论文单一事实源）。

## 2026-09-06 — 清理已作废的 PE 及探索性实验代码

论文已定调「诊断优先 + LOCO 校正算子」（决策日志 2026-09-02/03），
Prototype Expansion (PE) 方法作废。以下为该方法及其探索期的代码/输出，已删除。

### 保留的代码（当前方法，全部只读 `data2/`）

| 文件 | 作用 |
|---|---|
| `scripts/diagnostic_analysis.py` | 诊断核心，产出表 T1–T5 + `figures/Fig_diag_*.png` |
| `scripts/independence_diagnostic.py` | 条件独立性诊断，`diagnostic_analysis.py` 的配套 |
| `scripts/loco_correction_operator.py` | LOCO 留一城校正算子（Phase 1/2/2b/3） |
| `scripts/make_correction_maps.py` | 边界修正前/后可视化地图，复用 loco 模块 |
| `scripts/score_interannotator_recheck.py` | inter-annotator 盲标复核评分（TODO §6.8） |

### 已删除

**`scripts/` — 38 个旧实验脚本**
`bams_pe_ts_all_cities.py`, `bams_prototype_expansion.py`, `boundary_focused_experiments.py`,
`changsha_low_exclude_expand.py`, `city_confidence_bams_pe.py`, `city_random150_vs_bams_pe.py`,
`cross_city_second_ref_dw_esri.py`, `evaluate_city.py`, `generate_all_cities_comparison.py`,
`generate_changsha_comparison.py`, `hangzhou_bams100_pe.py`, `hangzhou_init_pipeline.py`,
`hefei_human_assist_wc.py`, `hefei_relabel_bams_pe.py`, `human_amplification_experiments.py`,
`margin_evaluation.py`, `nanjing_dual_eval_holdout.py`, `nanjing_error_top50_pe.py`,
`nanjing_high_3x3nb.py`, `nanjing_human_assist_wc.py`, `nanjing_nb8_experiment.py`,
`nanjing_nb8_protocol_v2.py`, `nanjing_phase1_diagnostics.py`, `nanjing_phase_b_rootcause.py`,
`nanjing_phase_c_sampling.py`, `nanjing_phase_d_expansion.py`, `nanjing_proto1_experiment.py`,
`nanjing_re1_nn8_experiment.py`, `nanjing_relabel_bams_pe.py`, `nanjing_round2_c12_pe.py`,
`plot_scene_frequency.py`, `plot_top100_accuracy.py`, `scheme_a_train_hard_correction.py`,
`scheme_d_conflict_downweight.py`, `seed_significance_experiments.py`, `unify_corrected_csvs.py`,
`wuhan_pipeline_step1_2.py`
（各脚本对应的实时输出目录仍保留在 `data2/`，只删代码。）

**notebook — PE 期探索 notebook**
`notebook/01_baseline.ipynb`, `notebook/02_wuhan.ipynb`, `notebook/03_changsha.ipynb`,
`notebook/04_hefei.ipynb`, `test.ipynb`（根目录）
（`notebook/` 里的 CSV / PNG 数据保留。）

## 2026-09-06 — 删除 `data/` 整个目录（旧版数据，已被 `data2/` 取代）

`data/` 是 7 月 PE 期的旧数据快照，已被 `data2/` 完整取代，当前所有脚本都不读它。

- **`data/{City}_WC_Samples_15000.csv` / `data/{City}_3x3_Features.csv`**（5–6 城）
  → 新版在 `data2/{City}/01_original/` 和 `data2/{City}/06_3x3/`，同为 15000 行、
  同样波段/指数特征，且新版多了 `Original_ID`（当前 pipeline 的连接键），旧版没有。
- **23 个 PE 期小 CSV**：`{City}_top100(_corrected).csv`, `{City}_scene.csv`,
  `{City}_Weight_Experiment.csv`, `Wuhan_Random100.csv`, `OA_Gain_Comparison.csv`,
  `FourCity_Replace_Results.csv`, `Random_Baseline_Results.csv`,
  `Table3_ErrorRate_vs_OAGain.csv`, `Table3_SceneFrequency.csv`,
  `Table4_SceneFrequency.csv`, `top100_corrected.csv`, `top100_uncertain_samples.csv`
  → 全是作废方法的中间产物；论文用的场景表是
  `data2/cross_city_scene_localization/bams150_scene_grouped.csv`。

原始数据资产的权威清单见 `docs/Paper_Writing_Plan.md` §5.3（全部指向 `data2/`）。

## 2026-09-06 — 清理 `data2/` 里已删脚本的实验产物

已删的 38 个脚本 + PE 期汇总产物在 `data2/` 下的输出目录/散文件，已删除。

### `data2/` 保留（15 项）

| 项 | 用途 |
|---|---|
| `Wuhan/ Hefei/ Nanchang/ Nanjing/ Changsha/ Hangzhou/` | 6 城主数据（01_original / 02_margin / 04_manual / 06_3x3 等，当前 pipeline 输入） |
| `cross_city_second_ref_dw_esri/` | DW/ESRI 第二参考，`diagnostic_analysis.py` + `loco_correction_operator.py` 读 |
| `cross_city_scene_localization/` | 场景分组表，`diagnostic_analysis.py` + `independence_diagnostic.py` 读 |
| `cross_city_aoi_season_balance/` | 6 城 AOI 季节/面积平衡核查（脚本不读，留作 Study Area 审稿备用） |
| `nanjing_baseline_autopsy/` | `VERDICT.md`，Discussion 引用（§5.3） |
| `nanjing_rootcause_v2/` | `ROOTCAUSE_STATUS.md`，Discussion 引用（§5.3） |
| `interannotator_recheck/` | 盲标复核，`score_interannotator_recheck.py` 读（TODO §6.8） |
| `diagnostic_analysis/ independence_diagnostic/ loco_correction_operator/` | 当前 3 个脚本的输出 |

### 已删除

**41 个实验文件夹**：`boundary_experiments/`, `human_amplification/`, `seed_experiments/`,
`scheme_a_hard_correction/`, `scheme_d_conflict_downweight/`,
`cross_city_all_tables/`, `cross_city_confidence_pe/`, `cross_city_random150_vs_bams/`,
`cross_city_story_alignment/`, `cross_city_wc_reference_bias/`（§6.2 已标注为过期表），
所有 `{city}_confidence_bams_pe/` / `{city}_random150_vs_bams/` / `{city}_relabel_bams_pe*/`
（changsha/hefei/nanchang/nanjing/wuhan），`changsha_low_exclude_expand/`,
`hefei_greenhouse_ablation/`, `hefei_human_assist_wc/`, `nanjing_human_assist_wc/`,
`hangzhou_bams100_pe/`, `hangzhou_bams150_pe/`, `nanjing_dual_eval_holdout/`,
`nanjing_error_top{50,100,150}_pe/`, `nanjing_high_3x3nb/`, `nanjing_high_teacher_student/`,
`nanjing_nb8_experiment/`, `nanjing_nb8_protocol_v2/`, `nanjing_phase1_diagnostics/`,
`nanjing_proto1_eval/`, `nanjing_random150/`, `nanjing_re1_nn8_eval/`,
`nanjing_round2_c12_pe/`（Round2 CSV 本体在 `Nanjing/04_manual/Nanjing_Round2_C12Boundary50.csv`，未动）

**25 个根目录散文件**：`AllCities_OA_pivot.{csv,xlsx}`, `AllCities_method_comparison.{csv,xlsx}`,
`all_city_{BAMS_PETS_results.csv,boundary_summary.xlsx,summary.xlsx}`, `BAMS_PETS_discussion.txt`,
`FourCity_Structure_Columns_Report.txt`, `boundary_error_results.csv`, `boxplot_results.png`,
`city_ranking.csv`, `discussion.txt`, `overall_{seed_summary,summary}.csv`,
`seed_{experiments_all.xlsx,results.csv,summary.csv}`, `significance_tests.csv`,
`statistical_significance_report.txt`, `winner_table.csv`,
`{cross_city_random150_vs_bams,hangzhou_bams100_pe,hangzhou_bams150_pe,three_city_confidence}_run.log`,
`~$进展报告_边界导向标注扩增.docx`

清理后 `data2/` 约 218 MB（几乎全是 6 城主数据）。已核对 5 个当前脚本的全部输入路径仍存在。

## 2026-09-06 — `data2/` 重新分类为 5 组顶层目录

清理后 `data2/` 直接摊了 15 个平级条目，重组为：

```
data2/
├── cities/              6 城主数据（Wuhan Hefei Nanchang Nanjing Changsha Hangzhou，各含 01–08 子目录）
├── shared_reference/    cross_city_second_ref_dw_esri/ + cross_city_scene_localization/ + cross_city_aoi_season_balance/
├── analysis_outputs/    diagnostic_analysis/ + independence_diagnostic/ + loco_correction_operator/（当前 3 个脚本的输出）
├── nanjing_rootcause/   nanjing_baseline_autopsy/ + nanjing_rootcause_v2/（Discussion 用）
└── interannotator_recheck/   标注盲标复核（自成一体，保持顶层不变）
```

子文件夹名一律保持不变，只在路径前插入一层分组目录。

**同步更新的代码/文档**（无 git，逐一手改并复跑验证）：
- `scripts/diagnostic_analysis.py`：`CITIES_DIR = DATA2/"cities"`；`REF_DIR = DATA2/"shared_reference"`；
  `OUT`/`LOCO` 移到 `analysis_outputs/`；docstring 路径同步。
- `scripts/independence_diagnostic.py`：`OUT` → `analysis_outputs/`；`SCENE_CSV` → `shared_reference/`。
- `scripts/loco_correction_operator.py`：`CITIES_DIR`；`OUT_DIR` → `analysis_outputs/`；`SECOND_REF` → `shared_reference/`；
  `load_city()` 的 `croot` 改用 `CITIES_DIR`；docstring 路径同步。
- `scripts/make_correction_maps.py`：无需改（复用 loco 模块的常量与 `load_city`）。
- `scripts/score_interannotator_recheck.py`：无需改（`interannotator_recheck/` 未移动）。
- `docs/Paper_Writing_Plan.md`：§2 加「`data2/` 目录结构」小节；§2 复跑命令注释、§3 Discussion、
  §5.2、§5.3 资产表、§6.2、§6.8 相关路径全部更新。

**验证**：`py_compile` 5 个脚本通过；`diagnostic_analysis.py` 与 `independence_diagnostic.py`
完整重跑，T1–T5 / S2–S3 数字与冻结值一致；`loco_correction_operator.load_city('Nanjing')`
+ `SECOND_REF.exists()` + `make_correction_maps` import 全部 OK。
