# Nanjing Phase B — Human vs WC seed contrast

- BAMS: n=150, high=73, agree/disagree=48/102

- Dual high pool: n=106, flip_vs_WC=69.8%, holdout=0.4


## Mean over seeds


Track           Method  n_seed_mean  WC_OA_mean  WC_OA_std    dWC_pp  WC_BE_mean  H_OA_mean     dH_pp  H_BE_mean  n_exp_mean
   WC         Baseline          0.0    0.909467   0.001039  0.000000       283.0        NaN       NaN        NaN         NaN
   WC     PE_Human_all        150.0    0.905289   0.001394 -0.417778       294.4        NaN       NaN        NaN        15.0
   WC    PE_Human_high         73.0    0.906489   0.000603 -0.297778       289.8        NaN       NaN        NaN        16.6
   WC        PE_WC_all        150.0    0.907689   0.001643 -0.177778       287.4        NaN       NaN        NaN        23.4
   WC       PE_WC_high         73.0    0.908178   0.001919 -0.128889       283.4        NaN       NaN        NaN        21.4
   WC     PE_Agree_all         48.0    0.906844   0.002004 -0.262222       286.6        NaN       NaN        NaN        12.4
   WC  PE_Disagree_all        102.0    0.905911   0.001431 -0.355556       291.8        NaN       NaN        NaN        17.2
   WC    PE_Agree_high         28.0    0.906622   0.001568 -0.284444       291.4        NaN       NaN        NaN         5.2
   WC PE_Disagree_high         45.0    0.907200   0.000990 -0.226667       288.8        NaN       NaN        NaN         8.2
 Dual    Baseline_dual          0.0    0.909467   0.001039  0.000000       283.0   0.665116  0.000000       13.6         NaN
 Dual PE_Human_holdout         63.0    0.908667   0.002066 -0.080000       282.0   0.609302 -5.581395       16.0        11.2
 Dual    PE_WC_holdout         63.0    0.908133   0.002189 -0.133333       284.4   0.586047 -7.906977       16.8        12.6


## Reading guide

- PE_WC_*: same BAMS positions, seed label = WorldCover (method can lift WC-OA?).

- PE_Human_*: seed label = human (passport-style).

- PE_Agree_* vs PE_Disagree_*: only WC-consistent / conflicting human seeds.

- Dual track: 40% high points held out for Human-OA; never used as seeds.

- H1 supported if PE_WC lifts WC-OA while PE_Human does not, and/or PE_Human lifts Human-OA while WC stays flat; and Disagree hurts WC while Agree helps.
