# BAMS150 Scene × Human–WC conflict (5 cities)

Source manuals: `data2/{City}/04_manual/{City}_BAMS150_Manual.csv`  
WC for Wuhan joined from `03_top150/*_BAMS150.csv` (`Class`); other cities from `Notes` (`WC_Class=`).

## Caveats
- Wuhan / Changsha / Nanchang use short Scene codes (BE, WE, RE, …).
- Hefei / Nanjing use free-text Chinese Scene notes → greenhouse / paddy / rural tags appear mainly there.
- Grouping is keyword-based for paper diagnostics, not a formal stratified sample.

## City-level human ≠ WC
| City | Disagree |
|------|----------|
| Nanjing | 68.0% |
| Wuhan | 58.0% |
| Changsha | 52.7% |
| Hefei | 49.3% |
| Nanchang | 44.7% |

## Localization-focused groups (n / disagree%)
| Group | Wuhan | Changsha | Nanchang | Hefei | Nanjing | ALL |
|-------|-------|----------|----------|-------|---------|-----|
| Greenhouse / polytunnel | — | — | — | 9/11% | — | 9/11% |
| Paddy / cropland / field | — | — | — | 9/44% | 7/43% | 16/44% |
| Rural settlement / surfaces | 1/100% | — | 6/33% | 20/35% | 12/50% | 39/41% |
| Water / water edge | 20/45% | 38/66% | 25/56% | 26/38% | 41/73% | 150/59% |
| Road / road edge | 29/52% | 13/69% | 20/50% | 19/68% | 25/80% | 106/63% |
| Bare land / construction | 12/33% | 11/9% | 23/43% | 9/44% | 11/55% | 66/38% |
| Built-up edge / urban fabric | 47/77% | 36/72% | 27/67% | 24/54% | 36/67% | 170/69% |
| Vegetation / forest edge | 8/38% | 34/41% | 7/57% | 33/67% | 15/73% | 97/56% |

Files: `bams150_scene_grouped.csv`, `scene_group_by_city.csv`
