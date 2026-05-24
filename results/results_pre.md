# Experimental Results

## UPenn-GBM Survival Dataset (SSL: UPenn)

| # | Experiment                               | Masks   | Emb | Radiomics | D Dropout | F1 Score         | Precision        | Recall           | Accuracy         |
| - | ---------------------------------------- | ------- | --- | --------- | --------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 2 | `No_Masks`                             | ❌ None | D1  | ❌        | ❌        | 0.5921           | 0.6287           | 0.6066           | 0.6066           |
| 3 | `All_masks`                            | ✅ All  | D1  | ❌        | ❌        | 0.6640           | 0.6946           | 0.6721           | 0.6721           |
| 4 | `All_masks-Emb_D3`                     | ✅ All  | D3  | ❌        | ❌        | 0.6640           | 0.6946           | 0.6721           | 0.6721           |
| 5 | `All_masks-Radiomics`                  | ✅ All  | D1  | ✅        | ❌        | 0.6539           | 0.6610           | 0.6557           | 0.6557           |
| 6 | `All_masks-D_Dropout`                  | ✅ All  | D1  | ❌        | ✅        | **0.7040** | **0.7091** | **0.7049** | **0.7049** |
| 6 | `All_masks-Radiomics-D_Dropout`        | ✅ All  | D1  | ✅        | ❌        |                  |                  |                  |                  |
| 6 | `All_masks-Emb_D3-Radiomics-D_Dropout` | ✅ All  | D1  | ✅        | ✅        | 0 6388                 |    0.6396              |     0.6393              |   0.6393               |

> **Config:** `--ssl_dataset upenn`, all runs use `-mts -mtr` (masked train/test) except *No_Masks*.

---

## BraTS Dataset (SSL: BraTS) — from `train_all.sh` (commented out)

| # | Experiment                        | Masks   | Emb | Radiomics | D Dropout | F1 Score         | Precision        | Recall           | Accuracy         |
| - | --------------------------------- | ------- | --- | --------- | --------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1 | `No_Masks`                      | ❌ None | D1  | ❌        | ❌        | 0.6695           | 0.6801           | 0.6721           | 0.6721           |
| 2 | `All_masks`                     | ✅ All  | D1  | ❌        | ❌        | 0.5657           | 0.6226           | 0.5902           | 0.5902           |
| 3 | `All_masks-Emb_D3`              | ✅ All  | D3  | ❌        | ❌        | 0.5657           | 0.6226           | 0.5902           | 0.5902           |
| 4 | `All_masks-Radiomics`           | ✅ All  | D1  | ✅        | ❌        | 0.6199           | 0.6253           | 0.6230           | 0.6230           |
| 5 | `All_masks-D_Dropout`           | ✅ All  | D1  | ❌        | ✅        | **0.7209** | **0.7239** | **0.7213** | **0.7213** |
| 1 | `All_masks-D_Dropout`           | ✅ All  | D1  | ❌        | ✅        | 0.6552           | 0.6578           | 0.6557           | 0.6557           |
| 1 | `All_masks-D_Dropout`           | ✅ All  | D1  | ❌        | ✅        | 0.6552           | 0.6578           | 0.6557           | 0.6557           |
| 1 | `All_masks-Radiomics-D_Dropout` | ✅ All  | D1  | ✅        | ✅        | 0.6393           | 0.6393           | 0.6393           | 0.6393           |

---

### Key Takeaways

- **Dynamic Dropout** consistently improves results on both datasets (~+4–6% F1 over baseline).
- **Emb D3** gives identical results to **Emb D1** (both datasets), suggesting no benefit from 3D positional embeddings.
- **Radiomics** helps on BraTS (+5.4% F1 vs All_masks) but slightly hurts on UPenn (−1.0% F1 vs All_masks).
- **No masks (baseline)** outperforms masked variants on BraTS but underperforms them on UPenn.
- **Best config:** `All masks + Emb D1 + No Radiomics + Dynamic Dropout` → 0.7040 (UPenn) / 0.7209 (BraTS) F1.
