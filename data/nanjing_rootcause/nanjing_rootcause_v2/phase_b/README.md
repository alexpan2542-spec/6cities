# Phase B — Human vs WC seed contrast (H1)

Script: `scripts/nanjing_phase_b_rootcause.py`

## Hypothesis
H1: Nanjing WC-OA stalls because human seed labels conflict with the WC exam — not because PE is broken.

## Design
- Same BAMS geometry; only seed labels change.
- WC-track: full BAMS seeds; K picked by WC (BE, −OA).
- Dual-track: high∪Round2, 60% seeds / 40% human hold-out.

## Key contrasts
1. `PE_Human_*` vs `PE_WC_*` (same positions)
2. `PE_Agree_*` vs `PE_Disagree_*`
3. Dual: Human-OA vs WC-OA for Human vs WC seeds
