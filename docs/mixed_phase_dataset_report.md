# Mixed-phase XRD 数据集生成报告

## 1. 模块定位

本阶段新增的是自动化高通量 XRD 表征平台中的 mixed-phase / multi-phase recognition 数据基础。它不是替代已有 single-phase multi-task CNN，而是在其基础上扩展到更接近真实样品的场景：

```text
真实合成产物或电池材料样品中常见主相 + 杂相 / 副产物 / 多相共存。
```

因此，混相模块的核心任务从 single-label classification 变为 multi-label phase identification：

```text
single-phase:
  XRD pattern -> one phase label

mixed-phase:
  XRD pattern -> multiple present phase labels
```

对应模型训练也应从 softmax + CrossEntropyLoss 转为 sigmoid + BCEWithLogitsLoss。

## 2. 数据路径

生成脚本：

```text
src/generate_mixed_phase_dataset.py
```

输入 single-phase 数据：

```text
data/processed/single_phase/
  train.npz
  val.npz
  test_normal.npz
  test_hard.npz
```

输出 mixed-phase 数据：

```text
data/processed/mixed_phase/mixed_v1/
  train.npz
  val.npz
  test_normal.npz
  test_hard.npz
  dataset_manifest.json
```

## 3. 数据规模

`mixed_v1` 与现有 single-phase split 保持相同样本数量，便于和单相模块形成并列实验。

| split | samples | X shape | single-phase | binary mixture | ternary mixture |
|---|---:|---:|---:|---:|---:|
| train | 31,050 | [31050, 3501] | 3,250 | 18,713 | 9,087 |
| val | 4,140 | [4140, 3501] | 429 | 2,526 | 1,185 |
| test_normal | 4,140 | [4140, 3501] | 410 | 2,511 | 1,219 |
| test_hard | 4,140 | [4140, 3501] | 428 | 2,460 | 1,252 |

总样本数：

```text
43,470 mixed-phase training/evaluation patterns
```

候选 phase labels：

```text
207 closed-set phases
```

XRD grid：

```text
2theta range: 10-90 degrees
length:       3501 points
wavelength:  Cu K-alpha, inherited from single-phase dataset
```

## 4. 生成策略

### 4.1 Split 隔离

混相生成严格保持 split 内部采样：

```text
mixed train       only uses single_phase/train.npz
mixed val         only uses single_phase/val.npz
mixed test_normal only uses single_phase/test_normal.npz
mixed test_hard   only uses single_phase/test_hard.npz
```

这样可以避免直接把 train pattern 混入 test pattern 的泄漏。每个 mixed sample 还保存 component source indices，便于后续追踪。

### 4.2 相数分布

默认采样策略：

```text
10% single-phase
60% binary mixtures
30% ternary mixtures
```

保留 single-phase 样本有两个目的：

1. 让 mixed-phase detector 学会“只有一个相存在”的情况，避免部署时强行报多个相。
2. 与单相模型形成桥接，使 workflow 可以在纯相和混相之间平滑切换。

第一版不生成四相或五相样本，因为相数过多会显著增加峰重叠和 minor phase detection 难度，不利于建立稳定 baseline。

### 4.3 混合系数

每个样本先随机选出 1-3 个 phase，再生成 synthetic intensity mixing coefficients。

主相系数范围：

```text
major coefficient: 0.60-0.95
```

最小可见系数：

```text
binary mixture:  each component >= 0.10
ternary mixture: each component >= 0.08
```

设置最小系数是为了避免某个 component 只有极弱贡献、几乎不可见，却仍被标注为 positive phase。脚本会先根据 `n_phases` 自动限制可采样的 major 上限：

```text
max_major_allowed = 1 - (n_phases - 1) * min_fraction
```

并在考虑 component intensity scale 后，确保 effective apparent fraction 仍满足最小可见阈值。当前数据中的最小非零 effective fraction 约为：

```text
train:       0.0800
val:         0.0801
test_normal: 0.0800
test_hard:   0.0800
```

注意：这里的 fraction 不应解释为真实质量分数。XRD 峰强还受结构因子、吸收、结晶度、择优取向等影响。因此报告中更准确的说法是：

```text
synthetic intensity mixing coefficient
```

或：

```text
apparent fraction in the synthetic mixture
```

当前同时保存两类 fraction：

```text
component_nominal_fractions:
  采样得到的 nominal synthetic mixing coefficient。

component_effective_fractions:
  nominal coefficient 乘以 component intensity scale 后重新归一化，
  更接近最终谱图中的 apparent intensity contribution。
```

`component_fractions` 与 `y_fraction` 使用 effective fraction，便于后续 fraction-level classification。

当前 split 的平均 effective 系数：

| split | mean major effective fraction | mean minor effective fraction |
|---|---:|---:|
| train | 0.7571 | 0.2044 |
| val | 0.7560 | 0.2063 |
| test_normal | 0.7549 | 0.2051 |
| test_hard | 0.7574 | 0.2023 |

当前 split 的平均 nominal 系数：

| split | mean major nominal fraction | mean minor nominal fraction |
|---|---:|---:|
| train | 0.7575 | 0.2041 |
| val | 0.7567 | 0.2058 |
| test_normal | 0.7553 | 0.2047 |
| test_hard | 0.7582 | 0.2017 |

### 4.4 光谱合成流程

当前生成流程为：

```text
1. 从对应 single-phase split 中抽取 component spectra
2. 每个 component 先做 max normalization
3. 按 synthetic intensity mixing coefficient 线性叠加
4. 对 mixed spectrum 施加全局 peak shift / background / noise
5. clip 到非负强度
6. 对最终 mixed spectrum 做 max normalization
```

默认参数：

```text
max_component_shift_bins = 0
max_global_shift_bins    = 1
background_range         = 0.00-0.02
noise_range              = 0.00-0.01
background_order         = 3
```

这里采用“混合后整体增强”为主，是因为真实样品通常由同一次仪器扫描获得，零点偏移、背景和噪声更接近全局扰动，而不是每个相拥有完全独立的仪器误差。

不过 component spectra 本身来自已增强的 single-phase split，因此仍保留了峰宽、强度扰动、背景和噪声等多样性。

## 5. 保存字段

每个 split `.npz` 保存 36 个字段：

```text
X
y_multi
y_presence
y_fraction
y_fraction_level
labels
two_theta
component_phase_indices
component_label_indices
component_source_indices
component_fractions
component_effective_fractions
component_nominal_fractions
component_mixing_coefficients
component_phase_labels
component_labels
component_shift_bins
component_intensity_scales
component_material_ids
component_record_ids
component_categories
component_subclasses
component_cif_paths
component_crystal_systems
component_spacegroup_symbols
component_spacegroup_numbers
global_shift_bins
background_levels
noise_levels
n_phases
num_phases
major_phase_index
major_fraction
source_split
generation_seed
mixing_seed
```

关键标签：

```text
y_presence:
  multi-hot phase presence label, shape = [N, 207]

y_fraction:
  effective apparent fraction vector, absent phase = 0

y_fraction_level:
  0 = absent
  1 = low     coefficient <= 0.25
  2 = medium  0.25 < coefficient <= 0.60
  3 = high    coefficient > 0.60
```

当前训练主任务仍是 `y_presence` 的 multi-label phase identification。新版 mixed CNN 训练脚本还支持一个轻量 fraction-level auxiliary head：只在真实存在的 phase 上预测 low / medium / high，用于帮助模型学习主相与副相的强度层级。该辅助任务不替代 presence 识别，只作为小权重正则项。

component metadata 字段继承自 single-phase split，用于后续 dashboard、case study 和 battery-relevant mixture 规则设计：

```text
component_material_ids
component_record_ids
component_categories
component_subclasses
component_cif_paths
component_crystal_systems
component_spacegroup_symbols
component_spacegroup_numbers
```

## 6. 与训练脚本的关系

混相训练脚本：

```text
src/train_mixed_phase_cnn.py
```

模型文件：

```text
src/model_mixture.py
```

推荐训练方式：

```powershell
python src\train_mixed_phase_cnn.py `
  --data-dir data\processed\mixed_phase\mixed_v1 `
  --run-name mixed_phase_cnn_v1_pretrained `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --pretrained-checkpoint models\residual_strong_lat003_ls005_v2_best_lattice\best_multitask_cnn.pt
```

训练脚本会在 validation set 上搜索 global threshold，并输出：

```text
models/mixed_phase/<run_name>/
results/mixed_phase/<run_name>/
logs/mixed_phase/<run_name>/
```

主要指标包括：

```text
micro precision / recall / F1
macro precision / recall / F1
subset accuracy
major phase accuracy
minor phase recall
top-k recall
average false positives per sample
mixture-size metrics
per-phase precision / recall / F1
fraction-level accuracy
```

## 7. 当前版本的合理性

`mixed_v1` 的目标是建立一个稳定、可复现、可训练的 baseline，而不是一次性覆盖所有真实复杂场景。

当前版本的优点：

1. 与 single-phase 数据完全对齐，phase label 空间一致。
2. train / val / test split 清晰，避免跨 split pattern 混用。
3. 支持 single / binary / ternary 三种相数。
4. 混合系数随机生成，并设有最小可见阈值。
5. 区分 nominal mixing coefficient 和 effective apparent fraction。
6. 保存 presence、apparent fraction、fraction level 三层标签。
7. 保存 component source、material metadata、global shift、background、noise 等追踪字段。
8. 训练目标保持简洁，以 multi-label phase identification 为主，并可用 fraction-level auxiliary loss 辅助主相 / 副相层级学习。

## 8. 当前限制

1. 当前组合仍以 closed-set random mixtures 为主，没有显式使用材料应用场景限制。
2. `test_hard` 继承了 single-phase hard split 的扰动来源，但还没有单独构造 high-overlap hard pairs。
3. 当前没有专门的 minor-only test，例如固定 90/10 或 85/15 主相 / 杂相。
4. 当前没有按 cosine similarity 选择谱图相似 phase pair。
5. 当前没有专门构造 same-formula different-spacegroup 的困难混相。
6. `y_fraction` 是 synthetic effective apparent coefficient，不等价于实验质量分数。
7. 当前暂不训练连续 fraction regression，因为该任务比 phase presence identification 更不稳定。

## 9. 下一步完善方向

### 短期

- 完成 `mixed_phase_cnn_v1_pretrained` 正式训练。
- 对 `test_normal` 与 `test_hard` 输出 micro/macro F1、major phase accuracy、minor phase recall。
- 比较 `from_scratch` 与 single-phase pretrained backbone 的差异。
- 根据 validation threshold calibration 分析 false positive / false negative trade-off。

### 中期

- 增加 `test_minor` split：

```text
major phase = 0.85-0.95
minor phase = 0.05-0.15
```

- 增加 `test_overlap` split：

```text
先计算 single-phase reference spectra 的 cosine similarity，
再选择 top similar pairs 构造高峰重叠混相。
```

- 增加 `test_same_formula_sg` split：

```text
same formula + different space group
```

用于检验当前 `formula + spacegroup` phase label 策略。

### 长期

- 构建 battery-relevant mixtures：

```text
cathode + decomposition product
anode + decomposition product
solid electrolyte + decomposition product
cathode + electrolyte + impurity
```

- 利用 metadata 中的 `sample_categories`、`sample_subclasses`、`sample_material_ids`、`sample_cif_paths` 进行更有材料语义的组合采样。
- 继续调优 fraction-level auxiliary classification：

```text
absent / low / medium / high
```

- 在 dashboard 中加入 per-phase recall、minor phase recall 和 co-occurrence error analysis。

## 10. 小结

`mixed_v1` 将项目从单相 XRD 表征扩展到了真实样品更常见的多相共存场景。当前设计重点是稳健 baseline：

```text
fixed synthetic mixed dataset
+ multi-label presence labels
+ clear split isolation
+ apparent mixing coefficient tracking
+ future-ready fraction level labels
```

后续应优先围绕 minor phase detection、high-overlap pairs 和 battery-relevant mixtures 建立更强的 hard evaluation，从而让混相模块更贴近 autonomous material characterization workflow。
