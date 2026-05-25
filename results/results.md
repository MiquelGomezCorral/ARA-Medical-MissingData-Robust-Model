# Experimental Results

## UPenn-GBM Survival Dataset (SSL: UPenn)

| # | Experiment                        | Masks   | Radiomics | F1 Score | Precision | Recall  | Accuracy |
| - | --------------------------------- | ------- | --------- | -------- | --------- | ------- | -------- |
| 1 | `No_Masks`                      | ❌ None  | ❌         | 0.5921   | 0.6287    | 0.6066  | 0.6066   |
| 1 | `No_Masks-Radiomics`            | ❌ None  | ✅         | 0.6392   | 0.6403    | 0.6393  | 0.6393   |
| 2 | `All_masks`                     | ✅ Train | ❌         | 0.6720   | 0.6732    | 0.6721  | 0.6721   |
| 3 | `All_masks-Radiomics`           | ✅ Train | ✅         | 0.6209   | 0.6274    | 0.6230  | 0.6230   |
| 2 | `All_masks`                     | ✅ All   | ❌         | 0.6850   | 0.7002    | 0.6885  | 0.6885   |
| 3 | `All_masks-Radiomics`           | ✅ All   | ✅         | 0.6850   | 0.7002    | 0.6885  | 0.6885   |

---

## BraTS Dataset (SSL: BraTS)

| # | Experiment                        | Masks   | Radiomics | F1 Score | Precision | Recall  | Accuracy |
| - | --------------------------------- | ------- | --------- | -------- | --------- | ------- | -------- |
| 1 | `No_Masks`                      | ❌ None  | ❌         | 0.5632   | 0.5851    | 0.5738  | 0.5738   |
| 1 | `No_Masks-Radiomics`            | ❌ None  | ✅         | 0.6066   | 0.6066    | 0.6066  | 0.6066   |
| 2 | `All_masks`                     | ✅ Train | ❌         | 0.5581   | 0.5903    | 0.5738  | 0.5738   |
| 3 | `All_masks-Radiomics`           | ✅ Train | ✅         | 0.6374   | 0.6410    | 0.6393  | 0.6393   |
| 2 | `All_masks`                     | ✅ All   | ❌         | 0.5795   | 0.6507    | 0.6066  | 0.6066   |
| 3 | `All_masks-Radiomics`           | ✅ All   | ✅         | 0.6546   | 0.6567    | 0.6557  | 0.6557   |
