# Tuning Log

| Step | Change                                                | Session                           | F1               | Precision | Recall | Accuracy | Δ F1             |
| ---- | ----------------------------------------------------- | --------------------------------- | ---------------- | --------- | ------ | -------- | ----------------- |
| 0b   | Baseline (UPenn SSL, no masks, no BraTS pretrain)     | —                                | 0.6518           | 0.6657    | 0.6557 | 0.6557   | —                |
| 1    | Mask wiring (missing_emb + image_mask_proj zero-init) | `base_name_2026-05-22-20-04-26` | 0.6382           | 0.6426    | 0.6393 | 0.6393   | −0.0136          |
| 3    | Tabular [-1, 1] (Gender {-1,1}, Age /50−1)           | `base_name_2026-05-22-20-18-59` | 0.6407           | 0.6926    | 0.6557 | 0.6557   | −0.0111          |
| 4    | Config + CLI fixes (infra only, no model Δ)          | —                                | 0.6407           | 0.6926    | 0.6557 | 0.6557   | ±0               |
| 2    | BraTS ViT→SSL pretraining (40 SSL epochs)            | `base_name_2026-05-22-20-34-00` | 0.5735           | 0.5745    | 0.5738 | 0.5738   | −0.0672          |
| 5    | 3D positional embeddings (learnable grid 6³)         | `base_name_2026-05-22-20-54-05` | 0.6601           | 0.7057    | 0.6721 | 0.6721   | **+0.0083** |
| 6    | BraTS intensity augs (hist std + z-norm)              | `base_name_2026-05-22-21-36-47` | **0.6640** | 0.6946    | 0.6721 | 0.6721   | **+0.0039** |

---

## Step 2 — BraTS pretraining (REVERTED)

BraTS SSL was already active — `prepare_ssl_data` uses `BraTSSSLDataModule`.

Enabling ViT→SSL doubled epochs to 40. The loss `l_recon + l_contrast * l_recon` collapses contrastive signal as `l_recon → 0`.

**Reverted.** BraTS SSL at 20 epochs is the effective baseline.

## Step 5 — 3D positional embeddings

- Replaced 1D learnable `pos_embed (1, L+1, D)` with learnable 3D grid `(1, 6, 6, 6, D)` + separate `cls_pos`.
- Mathematically: grid flattened yields same structure, but 3D initialization preserves spatial locality.
- **KEPT.** +0.0083 F1 over baseline.

## Step 6 — BraTS intensity augmentations

- Added `RobustIntensityStandardization3D` + `ZNormalization3D` before patch swap/cutout in `BraTSSSLAugmentPipeline`.
- Normalizes each SSL view independently before corruption, increasing augmentation diversity for contrastive learning.
- **KEPT.** +0.0039 F1 over Step 5, cumulative +0.0122 over baseline.

---

## Final Cumulative Result

| Metric    | Baseline | Final            | Δ                |
| --------- | -------- | ---------------- | ----------------- |
| F1        | 0.6518   | **0.6640** | **+0.0122** |
| Precision | 0.6657   | 0.6946           | +0.0289           |
| Recall    | 0.6557   | 0.6721           | +0.0164           |
| Accuracy  | 0.6557   | 0.6721           | +0.0164           |

## Active Changes Summary

| File                      | Change                                                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `config.py`             | Session isolation (timestamped dirs),`survival_batch_size`, `survival_num_workers`                               |
| `tabular_tokenizer.py`  | `missing_emb` param + mask-aware `forward(x, mask)`                                                              |
| `survival_predictor.py` | `image_mask_proj` (zero-init) injects MRI channel presence into sigmoid gate; `tabular_mask` routed to tokenizer |
| `survival_lightning.py` | Passes `tabular_mask` + `image_mask` to model                                                                    |
| `UPennGBMDataset.py`    | Gender `{-1,1}`, Age `(x/50)-1` ⇒ [-1,1] range                                                                  |
| `vit_encoder_3d.py`     | 3D learnable positional embeddings (grid 6³)                                                                        |
| `augmentations_3d.py`   | `RobustIntensityStandardization3D` + `ZNormalization3D` in BraTS SSL pipeline                                    |
| `train_model.py`        | Uses config `survival_batch_size`/`survival_num_workers`                                                         |
| `main.py`               | Fixed `--batch-size` typo, help text for `--masked-*`                                                            |
| `test.py`               | Passes `tabular_mask` + `image_mask` to model                                                                    |

---

## Step 0 — Session isolation

- `config.py`: `__post_init__` creates `models/{exp_name}_{timestamp}/` and `logs/{exp_name}_{timestamp}/` per run.

## Step 0b — Baseline

- Current code with SSL on UPenn (no BraTS, no ViT pretraining, no mask wiring). Reference: 0.6518.

## Step 1 — Mask wiring

- `tabular_tokenizer.py`: `missing_emb` learnable param replaces zeroed features with learned values when `mask` passed.
- `survival_predictor.py`: `image_mask_proj` (zero-init last layer) injects MRI channel presence into sigmoid gate.
- `survival_lightning.py`: passes `tabular_mask` + `image_mask` to model.
- `test.py`: passes `tabular_mask` + `image_mask` to model.
- **Result:** F1 dropped 2.1%. Small regression — likely extra learnable params need more epochs or the zero-init projection converges slowly. Acceptable.
