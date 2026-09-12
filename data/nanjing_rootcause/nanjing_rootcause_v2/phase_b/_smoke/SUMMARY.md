# Nanjing Phase B — Human vs WC seed contrast

- BAMS: n=150, high=73, agree/disagree=48/102

- Dual high pool: n=106, flip_vs_WC=69.8%, holdout=0.4


## Mean over seeds


Track           Method  n_seed_mean  WC_OA_mean  WC_OA_std    dWC_pp  WC_BE_mean  H_OA_mean     dH_pp  H_BE_mean  n_exp_mean
   WC         Baseline          0.0    0.910000        0.0  0.000000       277.0        NaN       NaN        NaN         NaN
   WC     PE_Human_all        150.0    0.908444        0.0 -0.155556       289.0        NaN       NaN        NaN        19.0
   WC    PE_Human_high         73.0    0.908889        0.0 -0.111111       279.0        NaN       NaN        NaN        14.0
   WC        PE_WC_all        150.0    0.910667        0.0  0.066667       278.0        NaN       NaN        NaN        29.0
   WC       PE_WC_high         73.0    0.908444        0.0 -0.155556       282.0        NaN       NaN        NaN        15.0
   WC     PE_Agree_all         48.0    0.909778        0.0 -0.022222       273.0        NaN       NaN        NaN        14.0
   WC  PE_Disagree_all        102.0    0.906889        0.0 -0.311111       287.0        NaN       NaN        NaN         9.0
   WC    PE_Agree_high         28.0    0.913111        0.0  0.311111       269.0        NaN       NaN        NaN         1.0
   WC PE_Disagree_high         45.0    0.911111        0.0  0.111111       277.0        NaN       NaN        NaN        14.0
 Dual    Baseline_dual          0.0    0.910000        0.0  0.000000       277.0   0.767442  0.000000       10.0         NaN
 Dual PE_Human_holdout         63.0    0.910667        0.0  0.066667       274.0   0.674419 -9.302326       14.0        23.0
 Dual    PE_WC_holdout         63.0    0.909778        0.0 -0.022222       281.0   0.674419 -9.302326       14.0        23.0


## Reading guide

- PE_WC_*: same BAMS positions, seed label = WorldCover (method can lift WC-OA?).

- PE_Human_*: seed label = human (passport-style).

- PE_Agree_* vs PE_Disagree_*: only WC-consistent / conflicting human seeds.

- Dual track: 40% high points held out for Human-OA; never used as seeds.

- H1 supported if PE_WC lifts WC-OA while PE_Human does not, and/or PE_Human lifts Human-OA while WC stays flat; and Disagree hurts WC while Agree helps.
