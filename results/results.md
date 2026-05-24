# Experimental Results

## UPenn-GBM Survival Dataset (SSL: UPenn)

| # | Experiment                        | Masks   | Radiomics | F1 Score         | Precision        | Recall           | Accuracy         |
| - | --------------------------------- | ------- | --------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1 | `No_Masks`                      | ❌ None | ❌         |           |           |           |           |
| 1 | `No_Masks-Radiomics`            | ❌ None | ✅         |           |           |           |           |
| 2 | `All_masks`                     | ✅ Train  | ❌         |           |           |           |           |
| 3 | `All_masks-Radiomics`           | ✅ Train  | ✅         |           |           |           |           |
| 2 | `All_masks`                     | ✅ All  | ❌         |           |           |           |           |
| 3 | `All_masks-Radiomics`           | ✅ All  | ✅         |           |           |           |           |
---

## BraTS Dataset (SSL: BraTS) — from `train_all.sh` (commented out)

| # | Experiment                        | Masks   | Radiomics | F1 Score         | Precision        | Recall           | Accuracy         |
| - | --------------------------------- | ------- | --------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1 | `No_Masks`                      | ❌ None | ❌         |           |           |           |           |
| 1 | `No_Masks-Radiomics`            | ❌ None | ✅         |           |           |           |           |
| 2 | `All_masks`                     | ✅ Train  | ❌         |           |           |           |           |
| 3 | `All_masks-Radiomics`           | ✅ Train  | ✅         |           |           |           |           |
| 2 | `All_masks`                     | ✅ All  | ❌         |           |           |           |           |
| 3 | `All_masks-Radiomics`           | ✅ All  | ✅         |           |           |           |           |