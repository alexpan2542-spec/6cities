# Nanjing Phase D — expansion ablation (H3)

- High seeds n=73; human≠WC=45 (61.6%)


## Mean over seeds (main methods)


           Pack        Method  OA_mean   OA_std  dOA_vs_RF_pp  dOA_vs_MLP_pp  BE_mean  dBE_vs_RF  n_seed_mean  n_exp_mean  agree_frac_mean
              —   Baseline_RF 0.903156 0.001741      0.000000      -0.631111    304.2        0.0          0.0         NaN              NaN
              —  Baseline_MLP 0.909467 0.001039      0.631111       0.000000    283.0      -21.2          0.0         NaN              NaN
BAMS_Human_high     SeedsOnly 0.903467 0.004435      0.031111      -0.600000    303.2       -1.0         73.0         0.0              NaN
BAMS_Human_high        PE_std 0.906489 0.000603      0.333333      -0.297778    289.8      -14.4         73.0        16.6         0.526554
BAMS_Human_high      PE_tight 0.907556 0.001023      0.440000      -0.191111    287.0      -17.2         73.0         6.2         0.652222
BAMS_Human_high    PE_nolimit 0.904444 0.001973      0.128889      -0.502222    290.2      -14.0         73.0       275.4         0.553602
BAMS_Human_high PE_expAgreeWC 0.906667 0.001993      0.351111      -0.280000    290.2      -14.0         73.0         9.6         0.492785
   BAMS_WC_high     SeedsOnly 0.905111 0.003446      0.195556      -0.435556    301.4       -2.8         73.0         0.0              NaN
   BAMS_WC_high        PE_std 0.908178 0.001919      0.502222      -0.128889    283.4      -20.8         73.0        21.4         0.406936
   BAMS_WC_high      PE_tight 0.907689 0.001631      0.453333      -0.177778    289.8      -14.4         73.0         6.6         0.380586
   BAMS_WC_high    PE_nolimit 0.904889 0.001927      0.173333      -0.457778    298.2       -6.0         73.0       237.2         0.373099
   BAMS_WC_high PE_expAgreeWC 0.906711 0.002620      0.355556      -0.275556    290.6      -13.6         73.0         8.4         0.468841


## Reading guide

- SeedsOnly ≥ PE_std → expansion wasted.

- SeedsOnly ≫ PE_std → expansion amplifies noise.

- PE_expAgreeWC best → conflict amplification in expanded labels.

- PE_tight vs PE_nolimit → margin gate usefulness.
