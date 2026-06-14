# Mixed-Phase Hard-Augmented Training Dataset Report

## Purpose

`mixed_v3_hard_augmented` is the first targeted training dataset after the `mixed_v2_hard_eval` diagnosis. It keeps the original `mixed_v1` random-mixture baseline data and adds hard samples only to the training and validation splits.

The goal is to test whether targeted data augmentation can improve:

- low-fraction minor phase recall
- high-overlap phase-pair recognition
- battery-relevant mixture robustness

without substantially reducing normal mixed-phase performance.

## Output Location

`data\processed\mixed_phase\mixed_v3_hard_augmented`

Generated files:

| File | Samples | Notes |
| --- | ---: | --- |
| `train.npz` | 43,050 | `mixed_v1/train` plus 12,000 hard samples |
| `val.npz` | 5,740 | `mixed_v1/val` plus 1,600 hard samples |
| `test_normal.npz` | 4,140 | copied from `mixed_v1/test_normal` |
| `test_hard.npz` | 4,140 | copied from `mixed_v1/test_hard` |
| `dataset_manifest.json` | - | generation metadata |

The independent hard benchmark remains:

`data\processed\mixed_phase\mixed_v2_hard_eval`

That benchmark should not be used for training.

## Augmentation Policy

Added hard samples:

| Split | Minor | Overlap | Battery-relevant | Total added |
| --- | ---: | ---: | ---: | ---: |
| train | 6,000 | 4,200 | 1,800 | 12,000 |
| val | 800 | 560 | 240 | 1,600 |

Proportions:

- 50% low-fraction minor-phase mixtures
- 35% high-overlap phase pairs
- 15% battery-relevant mixtures

Minor-phase nominal scenarios:

- 95/5
- 90/10
- 85/15
- 80/10/10
- 85/10/5

Overlap samples use cosine similarity among per-phase mean spectra and sample from the top 10% most similar phase pairs.

Battery-relevant scenarios:

- cathode + reference
- anode + reference
- solid electrolyte + reference
- cathode + solid electrolyte + reference
- anode + solid electrolyte + reference

## Split Composition

### train

| Group | Count |
| --- | ---: |
| baseline_random | 31,050 |
| minor_fraction | 6,000 |
| peak_overlap | 4,200 |
| battery_relevant | 1,800 |

Phase-count distribution:

| n phases | Count |
| ---: | ---: |
| 1 | 3,250 |
| 2 | 27,593 |
| 3 | 12,207 |

### val

| Group | Count |
| --- | ---: |
| baseline_random | 4,140 |
| minor_fraction | 800 |
| peak_overlap | 560 |
| battery_relevant | 240 |

Phase-count distribution:

| n phases | Count |
| ---: | ---: |
| 1 | 429 |
| 2 | 3,710 |
| 3 | 1,601 |

## Recommended First Training Run

The first v3 run should isolate the effect of hard augmentation, so it should use the presence-only mixed-phase CNN without the fraction auxiliary head.

```powershell
cd <project-root>
conda activate xrd-cnn

python src\train_mixed_phase_cnn.py `
  --data-dir data\processed\mixed_phase\mixed_v3_hard_augmented `
  --run-name mixed_phase_cnn_v3_hard_aug_presence_50ep `
  --epochs 50 `
  --batch-size 128 `
  --device auto `
  --pretrained-checkpoint models\residual_strong_lat003_ls005_v2_best_lattice\best_multitask_cnn.pt `
  --disable-fraction-level-head `
  --lambda-fraction-level 0 `
  --disable-weighted-sampling `
  --max-predictions 3
```

After training, evaluate the best checkpoint on `mixed_v2_hard_eval`:

```powershell
python src\evaluate_mixed_phase_cnn.py `
  --checkpoint models\mixed_phase\mixed_phase_cnn_v3_hard_aug_presence_50ep\best_mixed_phase_cnn.pt `
  --data-dir data\processed\mixed_phase\mixed_v2_hard_eval `
  --output-dir results\mixed_phase\hard_eval\v3_hard_aug_presence_on_mixed_v2_hard_eval `
  --threshold-mode all `
  --max-predictions 3 `
  --device auto `
  --batch-size 128 `
  --overwrite
```

## Success Criteria

The v3 run should be considered useful if it improves hard-benchmark robustness while preserving normal mixed-phase performance.

Primary targets:

- `test_minor` minor recall improves clearly over the v1 baseline
- `minor_5pct` and `minor_10pct` recall improve
- `test_overlap` micro-F1 does not regress
- `mixed_v3/test_normal` micro-F1 drops by less than about 0.01-0.02 compared with the v1 baseline

Baseline fixed-threshold reference from `mixed_v2_hard_eval`:

| Model | Split | micro-F1 | minor recall |
| --- | --- | ---: | ---: |
| v1 presence-only | test_minor | 0.644 | 0.209 |
| v1 presence-only | test_overlap | 0.788 | 0.585 |
| v1 presence-only | test_battery_relevant | 0.739 | 0.403 |
