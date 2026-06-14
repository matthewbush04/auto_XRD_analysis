# 数据集构建报告

## 1. 当前定位

本阶段对应 Word 文档中的 pilot 数据库建设，而不是一次性完成 500-1000 个材料的最终完整数据库。目标是先建立一个可复现、可训练、可扩展的 battery-material single-phase XRD 数据集。

当前不包含 electrocatalyst，属于 battery-focused Stage 1。后续如果要严格覆盖 Word 中“电池与电催化”完整范围，应增加 electrocatalyst profile。

## 2. 下载结果

新版 candidate-pool 下载逻辑运行后，当前数据库为：

```text
target_total:              300
candidate_count:           211
accepted material records: 211
phase labels:              207
download failures:         0
ordered structures:        211
```

按 category 统计：

```text
cathode:           111  (52.61%)
anode:              59  (27.96%)
solid_electrolyte:  30  (14.22%)
reference:          11  ( 5.21%)
```

按 subclass 统计：

```text
cathode/layered_oxide:             51
cathode/conversion_fluoride:       31
cathode/olivine_polyanion:         28
cathode/spinel_oxide:               1
anode/titanate_oxide:              28
anode/carbon_silicon_alloy:        24
anode/metal_reference:              7
solid_electrolyte/nasicon_oxide:   10
solid_electrolyte/sulfide:          9
solid_electrolyte/garnet_oxide:     7
solid_electrolyte/halide:           4
reference/common_decomposition:    11
```

数量没有达到 300 的主要原因是当前 battery-only `MATERIAL_SCOPE` 和 chemsys 列表下，唯一候选池本身只有 211 个；不是 API 大量失败造成的。

## 3. 下载策略

`download_materials.py` 采用两阶段策略：

```text
1. Collect candidates
2. Select balanced records
```

第一阶段按 category / subclass / chemsys 查询 Materials Project，默认每个 chemsys 最多保留：

```text
QUERY_TOP_K = 20
```

第二阶段按 category 和 subclass quota 选择材料。选择时优先考虑：

- stable structures。
- 更低的 `energy_above_hull`。
- 不同 `spacegroup` 的多样性。
- 每个 `phase_label` 最多 3 个 Materials Project variants。
- 默认要求 `structure.is_ordered == True`。

当前保存的主要 metadata 包括：

- `material_id`
- `formula_pretty`
- `phase_label`
- `category`
- `subclass`
- `chemical_system`
- `spacegroup_symbol`
- `spacegroup_number`
- `crystal_system`
- `energy_above_hull`
- `formation_energy_per_atom`
- `band_gap`
- `density`
- `volume`
- `nsites`
- `is_stable`
- `is_ordered`
- `cif_path`
- `structure_json_path`
- `query_chemsys`
- `structure_rank`
- `xrd_validation_status`
- `xrd_peak_count`
- `xrd_validation_error`

## 4. 标签策略

第一阶段分类标签为：

```text
phase_label = formula_pretty + "_" + spacegroup_symbol
```

`category` 和 `subclass` 是材料应用分类，只保存在 metadata 中，不作为单相 XRD phase identification 的直接标签。

这个策略可以区分明显不同的多晶型，但仍不能完美区分所有 ordering、site occupancy 或晶格常数差异。

## 5. 单相 XRD 数据集规模

已完成单相 `.npz` 数据生成：

```text
phase labels:    207
total patterns:  43,470
2theta range:    10-90 degrees
grid points:     3501
wavelength:      Cu K-alpha
```

split 规模：

```text
train:       31,050
val:          4,140
test_normal:  4,140
test_hard:    4,140
```

每个 phase label 的样本数为：

```text
train:       150
val:          20
test_normal:  20
test_hard:    20
```

已检查：

- `X` shape 正确。
- `X` 为 `float32`。
- 标签范围为 `0-206`。
- 所有谱图数值有限。
- 强度范围为 `0-1`。
- 没有全零谱。

现有 `.npz` 已补充晶系和空间群 metadata：

- `sample_phase_labels`
- `sample_crystal_systems`
- `crystal_system_y`
- `crystal_system_labels`
- `sample_spacegroup_symbols`
- `sample_spacegroup_numbers`

其中 `crystal_system_y` 支持直接训练 7-class crystal-system classification head。当前晶系标签为：

```text
Cubic
Hexagonal
Monoclinic
Orthorhombic
Tetragonal
Triclinic
Trigonal
```

## 6. XRD 生成流程

基础 peak pattern 使用 pymatgen：

```text
XRDCalculator(wavelength="CuKa").get_pattern(
    structure,
    two_theta_range=(10, 90)
)
```

2theta 网格为：

```text
t_i = 10 + i * (90 - 10) / (3501 - 1),  i = 0, ..., 3500
```

其中 `t_i` 是第 `i` 个 2theta 采样点。

## 7. Lattice Perturbation

为了模拟晶格常数轻微变化，对 `a,b,c` 做随机扰动：

```text
a' = a * (1 + eps_a)
b' = b * (1 + eps_b)
c' = c * (1 + eps_c)
```

其中：

```text
eps_a, eps_b, eps_c ~ Uniform(-0.005, 0.005)
```

`alpha,beta,gamma` 保持原值。扰动后的 `[a,b,c,alpha,beta,gamma]` 保存为 `lattice_params`，供后续 lattice-aware multi-task learning 使用。

## 8. Pseudo-Voigt Peak Profile

每个 Bragg peak 被展开为 pseudo-Voigt 峰形：

```text
G(t; mu, sigma) = exp(-0.5 * ((t - mu) / sigma)^2)
L(t; mu, sigma) = 1 / (1 + ((t - mu) / sigma)^2)
PV(t; mu, sigma, eta) = (1 - eta) * G(t; mu, sigma) + eta * L(t; mu, sigma)
```

其中：

- `mu` 是 pymatgen 生成的峰位。
- `sigma` 控制峰宽。
- `eta` 控制 Gaussian / Lorentzian 混合比例。

随机采样规则：

```text
sigma ~ Uniform(sigma_min, sigma_max)
eta   ~ Uniform(eta_min, eta_max)
```

## 9. Peak Intensity Jitter

为了模拟实验中相对峰强变化，每个峰乘以 log-normal jitter：

```text
j_k ~ LogNormal(mean=0, sigma=peak_intensity_jitter)
```

基础谱为：

```text
S_base(t) = normalize_max( sum_k I_k * j_k * PV(t; theta_k, sigma_k, eta_k) )
```

其中 `I_k` 是 pymatgen 给出的第 `k` 个峰强度。

## 10. Peak Shift

为模拟仪器零点误差或整体 peak position shift，谱图整体平移：

```text
delta ~ Uniform(-shift_range, shift_range)
S_shifted(t) = interp(t, t - delta, S_base)
```

当前范围：

```text
train / val / test_normal: shift_range = 0.03 degrees
test_hard:                 shift_range = 0.06 degrees
```

## 11. Intensity Scaling

整体强度缩放：

```text
s ~ Uniform(scale_min, scale_max)
S_scaled(t) = s * S_shifted(t)
```

当前范围：

```text
train / val / test_normal: s in [0.80, 1.20]
test_hard:                 s in [0.75, 1.25]
```

## 12. Polynomial Background

背景使用随机多项式生成。先定义：

```text
x_i = linspace(-1, 1, 3501)
c_m ~ Normal(0, 1)
c_0 = abs(c_0) + 0.5
B_raw(x) = sum_m c_m * x^m
```

然后归一化并缩放：

```text
B(x) = background_level * (B_raw - min(B_raw)) / max(B_raw - min(B_raw))
background_level ~ Uniform(background_min, background_max)
```

当前背景阶数：

```text
train / val / test_normal: order = 3
test_hard:                 order = 4
```

## 13. Gaussian White Noise

白噪声为：

```text
noise_level ~ Uniform(noise_min, noise_max)
epsilon_i ~ Normal(0, noise_level)
```

当前范围：

```text
train / val / test_normal: noise_level in [0.00, 0.03]
test_hard:                 noise_level in [0.03, 0.06]
```

## 14. 最终谱图

最终谱图为：

```text
X(t) = normalize_max( clip(S_scaled(t) + B(t) + epsilon(t), 0, +inf) )
```

因此所有 `X` 都被裁剪为非负，并按最大强度归一化到 `0-1`。

## 15. Bragg Metadata

每个样本保存 top-K peak metadata：

```text
TOP_K_PEAKS = 10
```

保存字段：

- `peak_hkls`
- `peak_2theta`
- `peak_mask`

这些字段用于后续：

- lattice-aware multi-task learning。
- Bragg peak error evaluation。
- optional Bragg-law physics consistency regularization。

## 16. 当前限制

1. 当前数据库仍是 battery-focused pilot，不是 Word 中 500-1000 材料的完整版本。
2. 当前不包含 electrocatalyst。
3. solid electrolyte、reference 和 spinel oxide 数量偏少。
4. 数据来自 Materials Project 计算结构，不等价于真实实验 PXRD。
5. 当前 peak width 是随机 `sigma`，不是完整 Caglioti `U,V,W` 模型。
6. preferred orientation 尚未显式建模，peak intensity jitter 只是弱近似。
7. test set 使用不同增强强度，但还不是 structure holdout。

## 17. 下一步建议

短期：

- 训练 single-phase classification baseline。
- 输出 accuracy、confusion matrix、per-class accuracy。

中期：

- 增加 lattice regression head，优先预测 `a,b,c`。
- 增加 crystal-system classification head，预测 cubic / hexagonal / monoclinic 等晶系。
- 输出 lattice MAE 和 Bragg peak error。
- 先把 Bragg consistency 作为 evaluation metric，再小权重加入 loss。

长期：

- 增加 mixed-phase synthetic generation。
- 将模型从 softmax single-label 扩展到 sigmoid multi-label。
- 加入 active learning uncertainty sampling。
