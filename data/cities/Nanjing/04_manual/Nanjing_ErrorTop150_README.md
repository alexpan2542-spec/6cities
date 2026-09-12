# Nanjing_ErrorTop150

Fill **Human_Class** (1/2/3) and **Confidence** (h/m/l). Optional: Scene.

## Selection
- RF(500) with **OOB** predictions on the standard **train** split (not in-sample; in-sample RF has ~0 train error)
- Keep `OOB_Pred ≠ WC_Class`, exclude BAMS150 ∪ Round2
- Prefer low OOB margin; quotas: C2→C1 55, C2→C3 45, C1→C2 35, then other errors
- n=150; all in train → OK for later PE without selecting on test

## error_type counts in file
```
error_type
C2_to_C1    70
C2_to_C3    45
C1_to_C2    35
```

File: `Nanjing_ErrorTop150.csv`
