# Nanjing re-labeled BAMS150 / PE experiment

- Human_Class changed vs backup: 29

- Confidence counts: {'high': 73, 'med': 60, 'low': 17}


## Mean over seeds

          Method  Mean_OA   Std_OA  Mean_BE    Std_BE  Mean_n_seed  Mean_n_exp  dOA_vs_Base_pp
        Baseline 0.903156 0.001946    304.2  5.932959          NaN         NaN        0.000000
     BAMS150_old 0.900444 0.001548    315.4  6.804410          NaN         NaN       -0.271111
     BAMS150_new 0.901022 0.000404    314.8  3.563706          NaN         NaN       -0.213333
BAMS150_new_high 0.902400 0.000507    309.0  2.915476          NaN         NaN       -0.075556
  BAMS150_new_hm 0.900711 0.000575    315.4  4.159327          NaN         NaN       -0.244444
          PE_old 0.906711 0.002219    286.2  7.463243        150.0        19.8        0.355556
          PE_new 0.905289 0.001559    294.4  8.876936        150.0        15.0        0.213333
     PE_new_high 0.906489 0.000674    289.8  5.403702         73.0        16.6        0.333333
       PE_new_hm 0.904844 0.002861    294.4 10.830512        133.0        21.4        0.168889


## Reading guide

- BAMS150_new_high: only Confidence=high overwrites WC; low/med keep WC on those points

- BAMS150_new_hm: high+med overwrite; low keep WC

- PE_* uses corresponding seed set for prototype expansion
