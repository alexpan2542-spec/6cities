# Nanjing Phase C — sampling contrast (H2)

- BAMS n=150 high=73; budget=150

- All PE_*_WC use WorldCover labels on the selected positions.

- PE_BAMS_Human_high is the passport reference only.


## Mean over seeds


            Method  OA_mean   OA_std  dOA_vs_RF_pp  dOA_vs_MLP_pp  BE_mean  dBE_vs_RF  dBE_vs_MLP  n_seed_mean  n_exp_mean
       Baseline_RF 0.903156 0.001741      0.000000      -0.631111    304.2        0.0        21.2          0.0         NaN
      Baseline_MLP 0.909467 0.001039      0.631111       0.000000    283.0      -21.2         0.0          0.0         NaN
PE_BAMS_Human_high 0.906489 0.000603      0.333333      -0.297778    289.8      -14.4         6.8         73.0        16.6
    PE_BAMS_WC_all 0.907689 0.001643      0.453333      -0.177778    287.4      -16.8         4.4        150.0        23.4
   PE_BAMS_WC_high 0.908178 0.001919      0.502222      -0.128889    283.4      -20.8         0.4         73.0        21.4
   PE_Random150_WC 0.908356 0.001338      0.520000      -0.111111    281.4      -22.8        -1.6        150.0        14.8
   PE_Margin150_WC 0.908756 0.002157      0.560000      -0.071111    284.6      -19.6         1.6        150.0        41.0
     PE_Easy150_WC 0.905689 0.001581      0.253333      -0.377778    289.8      -14.4         6.8        150.0         0.2


## Reading guide

- H2 (BAMS locations toxic): Random_WC / Easy_WC ≫ BAMS_WC.

- City-hard / PE weak: all WC samplers stuck near ~0–0.5 pp vs RF.

- Margin≈BAMS: uncertainty sampling same regime as BAMS.
