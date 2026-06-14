# Multi-task 1D-CNN 构建报告

## 1. 模型定位

本阶段实现的是自动化 XRD 表征平台中的核心 AI 算法模块。模型输入为单相合成 XRD pattern，输出三个互补结果：

1. `phase_label` 物相识别，207 类分类。
2. `crystal_system` 晶系识别，7 类分类。
3. `a,b,c` 晶格常数回归，单位为 angstrom。

该设计不是单一黑箱分类器，而是 physics-guided multi-task 1D-CNN：模型在识别物相的同时学习晶系和晶格参数，并通过 Bragg 定律评价预测结果的物理一致性。

## 2. 输入与数据

默认数据目录：

```text
data/processed/single_phase
```

训练脚本使用以下 split：

```text
train.npz
val.npz
test_normal.npz
test_hard.npz
```

关键输入字段：

```text
X:                  XRD intensity, shape = [N, 3501]
y:                  phase label index, 207 classes
crystal_system_y:   crystal system index, 7 classes
lattice_params:     [a,b,c,alpha,beta,gamma]
peak_hkls:          top-10 peak hkl labels
peak_2theta:        top-10 peak positions
peak_mask:          valid peak mask
```

模型内部将 `X` reshape 为：

```text
[batch_size, 1, 3501]
```

## 3. CNN 结构参数

代码文件：

```text
src/model_multitask.py
```

模型类：

```text
MultiTaskXRDConvNet
```

Backbone 参数：

```text
ConvBlock 1: Conv1d(1, 32, kernel_size=11)  + BatchNorm + GELU + MaxPool1d(2)
ConvBlock 2: Conv1d(32, 64, kernel_size=9)  + BatchNorm + GELU + MaxPool1d(2)
ConvBlock 3: Conv1d(64, 128, kernel_size=7) + BatchNorm + GELU + MaxPool1d(2)
ConvBlock 4: Conv1d(128,128, kernel_size=5) + BatchNorm + GELU + MaxPool1d(2)
```

位置特征保留：

```text
AdaptiveAvgPool1d(32)
```

这里没有使用 `AdaptiveAvgPool1d(1)`，因为 XRD 任务中峰位信息非常关键。池化到 32 个 bins 可以压缩特征长度，同时保留粗粒度 peak-position 信息。

Flatten 后特征维度：

```text
128 channels * 32 bins = 4096
```

共享投影层：

```text
Linear(4096, 512)
LayerNorm(512)
GELU
Dropout(0.20)
```

三个 head：

```text
phase_head:   Linear(512, 207)
crystal_head: Linear(512, 7)
lattice_head: Linear(512, 256) + GELU + Dropout(0.20) + Linear(256, 3)
```

训练脚本还支持可选增强版 lattice head：

```text
--strong-lattice-head
```

开启后 lattice head 变为：

```text
Linear(512, 256) + GELU + Dropout + Linear(256, 128) + GELU + Dropout + Linear(128, 3)
```

默认不强制开启，以保持旧 checkpoint 兼容。

新版还支持 phase-specific relative residual lattice prediction：

```text
--lattice-mode phase_residual
```

该模式不再直接预测绝对 `a,b,c`，而是先从 train set 统计每个 phase 的平均晶格参数：

```text
mean_abc[phase]
```

再让 CNN 预测相对残差：

```text
abc_pred = mean_abc[predicted_phase_distribution] * (1 + residual_scale * tanh(delta))
```

默认：

```text
residual_scale = 0.02
```

这表示模型最多在 phase-specific mean 附近预测约 ±2% 的相对偏移，比数据构建中的 ±0.5% 晶格扰动更宽，给 hard condition 和结构变体留出空间。

为了避免 lattice loss 反向破坏 phase classifier，phase probability 默认会 detach 后再送入 lattice head：

```text
phase_prob = softmax(phase_logits).detach()
```

## 4. Loss 设计

训练脚本：

```text
src/train_multitask_cnn.py
```

默认 loss：

```text
L = CE_phase + 0.3 * CE_crystal + 0.01 * MSE_lattice
```

其中：

```text
CE_phase:    phase_label cross entropy
CE_crystal:  crystal_system cross entropy
MSE_lattice: normalized a,b,c regression MSE
```

`a,b,c` 使用 train set 的均值和标准差标准化：

```text
target_norm = (target_abc - train_mean_abc) / train_std_abc
```

默认参数：

```text
lambda_sys  = 0.30
lambda_lat  = 0.01
lambda_phys = 0.0
```

Best checkpoint 默认按 validation phase accuracy 选择。这与项目主任务一致：自动物相识别是核心指标，晶格和晶系作为辅助任务提高结构理解能力。

在 `phase_residual` 模式下，lattice loss 使用相对误差的 SmoothL1 loss：

```text
SmoothL1((abc_pred - abc_true) / abc_true)
```

这样优化目标更接近相对晶格误差，而不是直接优化 Å² 尺度的绝对 MSE。

如果使用二阶段 lattice fine-tuning：

```text
--fine-tune-mode lattice_head
--fine-tune-mode lattice_late
```

best checkpoint 会自动改为按 validation lattice MAE 最小保存。

## 5. Physics-informed 模块

代码文件：

```text
src/physics.py
```

模型默认不强行把 Bragg loss 加入主训练目标，而是采用更稳定的 physics-informed reliability design：

1. lattice head 预测 `a,b,c`，这是与 XRD 峰位机制直接相关的物理量。
2. 使用预测 `a,b,c` 和真实 `alpha,beta,gamma` 重建晶格参数。
3. 使用 Bragg law 从 `hkl` 计算理论 `2theta`。
4. 与数据集中保存的 `peak_2theta` 比较，得到 Bragg peak mean absolute error。

Bragg consistency metric：

```text
mean |2theta_predicted_from_lattice - 2theta_reference|
```

该指标保存到：

```text
results/multitask_cnn/metrics_summary.json
```

训练脚本仍保留可选参数：

```text
--lambda-phys
```

默认值为 0。如果课程报告需要 ablation，可以尝试：

```text
--lambda-phys 1e-5
--lambda-phys 1e-4
```

不默认开启的原因：在 noisy / shifted synthetic XRD 条件下，直接把 Bragg loss 反传到共享 backbone 可能过度约束分类特征，导致 phase accuracy 下降。当前方案把 Bragg law 作为物理一致性与可靠性量化模块，更符合项目要求中的 error quantification 和 model reliability。

## 6. 训练与评估输出

默认输出目录：

```text
models/multitask_cnn
results/multitask_cnn
logs/multitask_cnn
```

主要文件：

```text
models/multitask_cnn/best_multitask_cnn.pt
results/multitask_cnn/run_config.json
results/multitask_cnn/training_history.json
results/multitask_cnn/training_curves.png
results/multitask_cnn/metrics_summary.json
logs/multitask_cnn/training_log.jsonl
```

新版训练脚本支持 `--run-name`，默认不会覆盖已有实验目录。若未指定 `--run-name`，脚本会自动创建带时间戳的目录：

```text
models/multitask_cnn_YYYYMMDD_HHMMSS
results/multitask_cnn_YYYYMMDD_HHMMSS
logs/multitask_cnn_YYYYMMDD_HHMMSS
```

如果指定：

```text
--run-name lambda_lat_005
```

输出目录为：

```text
models/lambda_lat_005
results/lambda_lat_005
logs/lambda_lat_005
```

若同名目录已存在且有文件，脚本会拒绝覆盖。只有显式传入 `--overwrite-run` 才会复用同名目录。

每个 evaluated split 会保存：

```text
*_phase_confusion_matrix.csv
*_phase_confusion_matrix.png
*_phase_per_class_accuracy.csv
*_phase_top_confusions.csv
*_crystal_confusion_matrix.csv
*_crystal_confusion_matrix.png
*_crystal_per_class_accuracy.csv
*_predictions.csv
```

`*_predictions.csv` 包括：

```text
true/pred phase label
phase confidence
true/pred crystal system
crystal confidence
true/pred a,b,c
top-3 / top-5 hit flags
```

在 `phase_residual` 模式下，`metrics_summary.json` 会额外保存 oracle 上限指标：

```text
oracle_true_phase_lattice_mae_mean
oracle_true_phase_bragg_peak_mae_deg
```

主指标仍然使用 predicted phase distribution，代表可部署自动流程；oracle 指标使用 true phase mean，只用于分析“若物相识别完全正确，晶格 refinement 的理论上限”。

这些文件可以直接用于后续 dashboard 或课程展示。

## 7. 推荐训练命令

快速 smoke test：

```powershell
python src\train_multitask_cnn.py `
  --epochs 1 `
  --batch-size 64 `
  --max-train-samples 512 `
  --max-val-samples 128 `
  --max-test-samples 128 `
  --device auto
```

正式 GPU baseline：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name baseline `
  --epochs 30 `
  --batch-size 128 `
  --device auto
```

提高 lattice loss 权重：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name lambda_lat_005 `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --lambda-lat 0.05
```

Label smoothing 实验：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name label_smoothing_005 `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --label-smoothing 0.05
```

Phase-specific residual lattice 实验：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name residual_label_smoothing_005 `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --label-smoothing 0.05 `
  --lattice-mode phase_residual `
  --residual-scale 0.02
```

Residual scale ablation：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name residual_scale_001 `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --label-smoothing 0.05 `
  --lattice-mode phase_residual `
  --residual-scale 0.01
```

二阶段 lattice head refinement：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name stage2_lattice_head `
  --resume-checkpoint models\multitask_cnn\best_multitask_cnn.pt `
  --fine-tune-mode lattice_head `
  --epochs 10 `
  --batch-size 128 `
  --lr 1e-4 `
  --lambda-lat 1.0 `
  --device auto
```

二阶段 late-backbone refinement：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name stage2_lattice_late `
  --resume-checkpoint models\multitask_cnn\best_multitask_cnn.pt `
  --fine-tune-mode lattice_late `
  --epochs 10 `
  --batch-size 128 `
  --lr 1e-4 `
  --lambda-lat 0.1 `
  --device auto
```

可选 physics ablation：

```powershell
& "PATH_TO_GPU_ENV\python.exe" src\train_multitask_cnn.py `
  --run-name lambda_phys_1e5 `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --lambda-lat 0.05 `
  --lambda-phys 1e-5
```

## 8. 创新点总结

1. 多任务学习：同一个 XRD backbone 同时服务物相识别、晶系识别和晶格常数回归。
2. 位置保留式 pooling：使用 `AdaptiveAvgPool1d(32)` 保留峰位信息，避免全局池化完全抹掉 diffraction peak 的空间位置。
3. 物理辅助监督：`a,b,c` 回归让模型学习与 XRD 峰位形成机制直接相关的结构参数。
4. Bragg-law reliability metric：使用 Bragg 定律量化预测晶格参数的物理一致性，作为模型可靠性评估。
5. Robust evaluation：同时输出 `test_normal` 和 `test_hard` 指标，用 hard split 检查模型对峰位偏移、噪声、背景和峰宽变化的鲁棒性。
6. 可视化接口准备：训练脚本保存 confusion matrix、per-class accuracy、top confusions、prediction CSV 和 training curves，后续 dashboard 可以直接读取。
7. 二阶段晶格精修：支持从最佳物相识别模型继续训练 lattice head 或 late backbone，以降低晶格参数误差，同时尽量保持 phase accuracy。
8. 实验防覆盖：通过 `--run-name` 和默认拒绝覆盖机制，避免多组实验互相污染。
9. Phase-specific residual lattice prediction：将绝对晶格回归转化为“phase 先验均值 + 相对残差”预测，更符合当前数据集中晶格扰动的物理生成机制。

## 9. 当前实现状态

已完成：

```text
src/model_multitask.py
src/physics.py
src/train_multitask_cnn.py
check_environment.py
docs/cnn_model_report.md
```

已通过小规模 smoke test：

```text
epochs = 1
train samples = 64
val samples = 32
test_normal samples = 32
test_hard samples = 32
device = cpu
```

该 smoke test 只验证工程链路，不代表正式模型精度。正式结果需要在 GPU 环境上使用完整数据集训练。
