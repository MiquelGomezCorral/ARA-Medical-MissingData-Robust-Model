# AGENTS.md

## Quick Start

**Setup:**
```bash
conda create --name ARA_env python=3.13 -y
conda activate ARA_env
uv pip install -r requirements.txt
pip install -e .
```

**Run training:**
```bash
python app/main.py train [OPTIONS]
```

## Architecture

- **Entry point:** `app/main.py` - CLI with subcommands
- **Source package:** `app/src/` (editable install)
- **Scripts:** `app/scripts/` - training orchestration
- **Configs:** `configs/*.yaml` loaded via `--config` arg

**Training pipeline (3 stages):**
1. SSL pretraining → `app/src/training/train_ssl.py:train_stage_ssl()`
2. Survival training → `app/src/training/train_survival.py:train_stage_survival()`
3. Testing → `app/src/training/test.py`

**Data:**
- BraTS dataset (SSL): `data/BraTS/`
- UPenn-GBM (survival): `data/MR_NIfTI/`
- Partition IDs: `data/partitions_ids.json`

## Key Commands

| Task | Command |
|------|---------|
| Full training | `python app/main.py train` |
| SSL only | `python app/main.py train --ssl_epochs 50` |
| Survival only | Modify `app/scripts/train_model.py` to skip stages |
| Custom config | `python app/main.py train --config custom.yaml` |

## Important Conventions

- **Checkpoints:** Saved to `models/` directory
- **Logs:** Written to `logs/`
- **Data paths:** Relative to `app/src/` (see `config.py:18-47`)
- **Seeding:** `torch`, `numpy`, `random` via `src/utils/seed.py`
- **Accelerator:** Auto-detects GPU via `src/training/device.py`

## Config System

`app/src/config/config.py:Configuration` dataclass:
- Loads YAML from `configs/` via `--config` flag
- Default values in dataclass, YAML overrides
- Creates directories on `__post_init__`

## Common Gotchas

- **Data not ignored:** `data/*` in `.gitignore` - do not commit datasets
- **Editable install required:** `pip install -e .` for `src/` imports
- **CUDA:** Enabled by default (`cuda: True` in config)
- **Early stopping:** Enabled by default; disable with `--no_early_stopping`

## Testing

No formal test suite. Verify via training run:
```bash
python app/main.py train --ssl_epochs 1 --survival_epochs 1
```


