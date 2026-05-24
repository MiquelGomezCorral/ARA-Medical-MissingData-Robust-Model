# Experiment Log

Non-destructive experiments. Run order: Phase 0 → 1 → 2 → 3a → 3b → 3c.

**Baseline runs:** `--ssl_epochs 100 --survival_epochs 50`
**Experiment runs:** `--ssl_epochs 30 --survival_epochs 15` (faster screening)
**All runs:** `--ssl_num_workers 0 --ssl_batch_size 8 --num_workers 0 --survival-batch-size 4` (avoids DataLoader worker OOM)

## Parameters
- deterministic: False (cudnn benchmark=True, allows variance)
- SSL loss: l_recon + l_contrast * l_recon (unchanged)
- Data split: 80/10/10 (UPenn)
- Tabular: Gender {-1,1}, Age (x/50)-1
- 3D positional embeddings
- BraTS: RobustIntensityStandardization + ZNormalization in SSL pipeline
- Mask wiring: missing_emb + image_mask_proj (zero-init)

---

## Phase 1: New Baselines

### B1 — BraTS SSL, 50/25 epochs
| Metric | Value |
|--------|-------|
| SSL dataset | BraTS (1251) |
| Command | `train -mtr -mts --ssl_epochs 50 --survival_epochs 25 --ssl_dataset brats` |
| F1 | 0.5653 |
| Precision | 0.5773 |
| Recall | 0.5738 |
| Accuracy | 0.5738 |
| Workers | 0 (IPC overhead hurts more than helps) |
| Notes | Regression from 0.6640 (20/20). 50 SSL epochs + contrastive loss may collapse. |

### B2 — UPenn SSL, 50/25 epochs
| Metric | Value |
|--------|-------|
| SSL dataset | UPenn (630) |
| Command | `train -mtr -mts --ssl_epochs 50 --survival_epochs 25 --ssl_dataset upenn` |
| F1 | 0.6053 |
| Precision | 0.6093 |
| Recall | 0.6066 |
| Accuracy | 0.6066 |
| Workers | 12 (fast, no OOM) |
| Notes | Better than B1 BraTS SSL (0.5653) but below 0.6640. Early stop at surv epoch 13. |

---

## Phase 2: Revert Mask Wiring

Reverted: missing_emb in TabularTokenizer, image_mask_proj in SurvivalPredictor, mask forwarding in LightningModule/test.py.

### M1 — BraTS SSL, no mask wiring, 20/10 epochs
| Metric | Value |
|--------|-------|
| F1 | 0.5888 |
| Precision | 0.5904 |
| Recall | 0.5902 |
| Accuracy | 0.5902 |
| vs B1 Δ | +0.0235 (but B1 had 50/25 epochs) |
| Workers/Batch | 12/8 |
| Notes | batch_size=8 suboptimal; contrastive loss needs larger batches. |

### M2 — UPenn SSL, no mask wiring, 20/10 epochs
| Metric | Value |
|--------|-------|
| F1 | 0.6053 |
| Precision | 0.6093 |
| Recall | 0.6066 |
| Accuracy | 0.6066 |
| vs B2 Δ | 0.0000 (identical!) |
| Workers/Batch | 12/32 |
| Notes | Mask wiring has ZERO effect on UPenn SSL. Batch size 32 same as batch 8. |

---

## Phase 3a: Dynamic Dropout

Re-added `_generate_dynamic_dropout()` for train-time random missingness.

### D1 — BraTS SSL, no dynamic dropout, 20/10 epochs, batch 32
| Metric | Value |
|--------|-------|
| F1 | 0.6364 |
| Precision | 0.6461 |
| Recall | 0.6393 |
| Accuracy | 0.6393 |
| vs M1 Δ | +0.0476 (batch 32 + no dynamic dropout) |
| Notes | Batch size 32 critical for contrastive SSL. |

### D2 — UPenn SSL, no dynamic dropout, 20/10 epochs, batch 32
| Metric | Value |
|--------|-------|
| F1 | 0.6382 |
| Precision | 0.6426 |
| Recall | 0.6393 |
| Accuracy | 0.6393 |
| vs M2 Δ | +0.0329 (batch 32 + no dynamic dropout) |
| Notes | Dynamic dropout removal helps; batch 32 is key. |

---

## Phase 3b: 1D Positional Embeddings

Reverted 3D grid pos embed → 1D (as in b94f197 commit).

### P1 — BraTS SSL + 1D pos embed
| Metric | Value |
|--------|-------|
| F1 | |
| Precision | |
| Recall | |
| Accuracy | |
| vs best Δ | |
| Notes | |

### P2 — UPenn SSL + 1D pos embed
| Metric | Value |
|--------|-------|
| F1 | |
| Precision | |
| Recall | |
| Accuracy | |
| vs best Δ | |
| Notes | |

---

## Phase 3c: Augmentation Cross-Pollination

**BraTS Base Augs:** RobustIntensityStandardization + ZNormalization + PatchSwap + Cutout
**UPenn Base Augs:** RobustIntensityStandardization + ZNormalization + Flip + Rotate90 + IntensityScaling + GaussianNoise + RandomCrop

### A1 — BraTS SSL + UPenn safe augs
| Metric | Value |
|--------|-------|
| F1 | |
| Precision | |
| Recall | |
| Accuracy | |
| vs best Δ | |
| Notes | Added Flip, Rotate90 (p=0.5), IntensityScale, Noise (std=0.05), Crop (min_scale=0.85) to BraTS pipeline |

### A2 — UPenn SSL + BraTS patch/cutout augs
| Metric | Value |
|--------|-------|
| F1 | |
| Precision | |
| Recall | |
| Accuracy | |
| vs best Δ | |
| Notes | Added PatchSwap (p=0.8), Cutout (p=0.8) to UPenn pipeline |
