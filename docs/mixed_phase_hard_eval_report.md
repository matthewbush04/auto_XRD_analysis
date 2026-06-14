# Mixed-Phase Hard Evaluation Report

## Purpose

The original `mixed_v1` dataset is now treated as the random major-minor baseline for mixed-phase CNN training and normal evaluation. The new `mixed_v2_hard_eval` dataset is a targeted benchmark for diagnosing realistic failure modes:

- low-fraction minor phases
- highly overlapping phase pairs
- battery-relevant phase combinations

This benchmark is not intended to replace `mixed_v1`. It is designed to answer what the baseline model struggles with after it has already learned ordinary random mixtures.

## Generated Dataset

Output directory:

`data\processed\mixed_phase\mixed_v2_hard_eval`

Source single-phase split:

`data\processed\single_phase\test_hard.npz`

Generated files:

| File | Samples | Main purpose |
| --- | ---: | --- |
| `test_minor.npz` | 4,140 | Low-fraction impurity detection |
| `test_overlap.npz` | 4,140 | Peak-overlap sensitivity |
| `test_battery_relevant.npz` | 4,140 | Battery-scene transfer |

The generated arrays keep the same core schema as `mixed_v1`, including `X`, `y_multi`, `y_fraction_level`, `component_phase_indices`, `component_fractions`, `n_phases`, and `major_phase_index`. Additional diagnostic fields were added:

- `minor_fraction_bin`
- `hard_eval_type`
- `component_pair_similarity`
- `overlap_rank`
- `overlap_percentile`
- `overlap_bin`
- `battery_scenario`

## Dataset Composition

### test_minor

Nominal fraction scenarios:

- 95/5
- 90/10
- 85/15
- 80/10/10
- 85/10/5

Counts:

| Group | Count |
| --- | ---: |
| 2-phase | 2,484 |
| 3-phase | 1,656 |
| minor_5pct | 1,656 |
| minor_10pct | 1,656 |
| minor_15pct | 826 |

Mean effective major fraction: 0.870  
Mean effective minor fraction: 0.093

### test_overlap

Phase pairs were selected by cosine similarity among per-phase mean spectra. The test samples are evenly split across the top similarity bins:

| Overlap bin | Count |
| --- | ---: |
| top_1pct | 1,380 |
| top_5pct | 1,380 |
| top_10pct | 1,380 |

All samples are binary mixtures. Mean effective minor fraction is 0.200.

### test_battery_relevant

Battery-relevant scenarios:

| Scenario | Count |
| --- | ---: |
| anode_reference | 828 |
| anode_solid_electrolyte_reference | 828 |
| cathode_reference | 828 |
| cathode_solid_electrolyte_reference | 828 |
| solid_electrolyte_reference | 828 |

Counts:

| Group | Count |
| --- | ---: |
| 2-phase | 2,484 |
| 3-phase | 1,656 |

Mean effective major fraction: 0.810  
Mean effective minor fraction: 0.136

## First Hard-Eval Results

Evaluation settings:

- `max_predictions = 3`
- fixed threshold uses the checkpoint threshold
- val-calibrated threshold is recalibrated on `mixed_v1/val.npz`
- oracle threshold is selected on each hard-eval split for diagnosis only

Result directories:

- `results\mixed_phase\hard_eval\v1_presence_on_mixed_v2_hard_eval`
- `results\mixed_phase\hard_eval\v2_frac_aux_l005_on_mixed_v2_hard_eval`

### v1 presence-only baseline

| Split | Mode | Threshold | micro-F1 | minor recall | top3 all-hit |
| --- | --- | ---: | ---: | ---: | ---: |
| test_minor | fixed | 0.525 | 0.644 | 0.209 | 0.287 |
| test_minor | val_calibrated | 0.400 | 0.644 | 0.240 | 0.287 |
| test_minor | oracle | 0.475 | 0.645 | 0.221 | 0.287 |
| test_overlap | fixed | 0.525 | 0.788 | 0.585 | 0.735 |
| test_overlap | val_calibrated | 0.400 | 0.785 | 0.622 | 0.735 |
| test_overlap | oracle | 0.675 | 0.792 | 0.542 | 0.735 |
| test_battery_relevant | fixed | 0.525 | 0.739 | 0.403 | 0.495 |
| test_battery_relevant | val_calibrated | 0.400 | 0.743 | 0.444 | 0.495 |
| test_battery_relevant | oracle | 0.375 | 0.744 | 0.453 | 0.495 |

Key fixed-threshold subgroup results:

| Diagnostic group | micro-F1 | minor recall | top3 all-hit |
| --- | ---: | ---: | ---: |
| minor_5pct | 0.576 | 0.091 | 0.062 |
| minor_10pct | 0.647 | 0.231 | 0.308 |
| minor_15pct | 0.778 | 0.492 | 0.691 |
| overlap top_1pct | 0.739 | 0.572 | 0.690 |
| overlap top_5pct | 0.805 | 0.582 | 0.749 |
| overlap top_10pct | 0.824 | 0.602 | 0.766 |

### v2 fraction-aux l005 baseline

| Split | Mode | Threshold | micro-F1 | minor recall | top3 all-hit |
| --- | --- | ---: | ---: | ---: | ---: |
| test_minor | fixed | 0.600 | 0.640 | 0.200 | 0.281 |
| test_minor | val_calibrated | 0.550 | 0.641 | 0.212 | 0.281 |
| test_minor | oracle | 0.500 | 0.642 | 0.225 | 0.281 |
| test_overlap | fixed | 0.600 | 0.786 | 0.575 | 0.728 |
| test_overlap | val_calibrated | 0.550 | 0.784 | 0.590 | 0.728 |
| test_overlap | oracle | 0.700 | 0.789 | 0.544 | 0.728 |
| test_battery_relevant | fixed | 0.600 | 0.734 | 0.389 | 0.494 |
| test_battery_relevant | val_calibrated | 0.550 | 0.738 | 0.409 | 0.494 |
| test_battery_relevant | oracle | 0.450 | 0.740 | 0.446 | 0.494 |

## Interpretation

The hard benchmark confirms that the main weakness is low-fraction minor phase detection, especially near 5 percent. Lowering the threshold increases minor recall slightly but does not solve the problem, and oracle thresholding barely improves `test_minor` micro-F1. This suggests the issue is not only calibration; the model often does not rank the weak minor phase high enough.

The overlap benchmark is difficult but less severe than the 5 percent minor benchmark. Performance degrades most in the `top_1pct` overlap bin, which is expected and useful for reporting.

The fraction-level auxiliary model does not improve hard phase detection in this run. It remains useful as an ablation, but the presence-only model should stay the main mixed-phase phase-identification baseline.

## Recommended Next Step

The next high-value experiment is a separate hard-augmented training dataset, not another small model tweak. It should add targeted samples for:

- 5-10 percent minor phases
- high-overlap pairs from the top 1-5 percent similarity bins
- 3-phase battery-relevant mixtures with a weak reference/decomposition component

The success criterion should be improved `test_minor` minor recall and `test_overlap` micro-F1 while keeping `mixed_v1 test_normal` degradation below about 0.01-0.02 micro-F1.
