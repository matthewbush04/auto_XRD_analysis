# 单相多任务 CNN 简要报告

## 1. 模型目标

单相模型用于从一维 XRD 谱图中同时预测：

| 任务 | 输出 |
| --- | --- |
| 物相识别 | 207 类 `phase_label` |
| 晶系识别 | 7 类 `crystal_system` |
| 晶格参数回归 | `a,b,c` |

主任务是物相识别，晶系和晶格参数作为辅助任务，用于增强结构理解和物理一致性分析。

## 2. 输入数据

数据目录：`data/processed/single_phase`

| 文件 | 用途 |
| --- | --- |
| `train.npz` | 训练 |
| `val.npz` | 验证 |
| `test_normal.npz` | 常规测试 |
| `test_hard.npz` | 强扰动测试 |

每个样本的谱图长度为 3501，对应 10-90 度 2θ 范围。

## 3. 模型结构

模型文件：`src/model.py`

结构概括：

- 一维卷积主干提取峰位、峰形和背景特征。
- 全局池化得到谱图表示。
- 三个输出头分别负责物相分类、晶系分类和晶格参数回归。

模型支持两种晶格回归方式：

| 模式 | 说明 |
| --- | --- |
| 直接回归 | 直接预测 `a,b,c` |
| 相对残差回归 | 以每个物相的平均晶格参数为先验，预测小范围相对残差 |

相对残差模式更符合当前数据集中晶格扰动较小的生成方式。

## 4. 损失函数

默认训练目标：

```text
L = CE_phase + 0.3 * CE_crystal + 0.01 * MSE_lattice
```

含义：

- `CE_phase`：物相分类损失；
- `CE_crystal`：晶系分类损失；
- `MSE_lattice`：归一化晶格参数回归损失。

最佳 checkpoint 默认按验证集物相准确率保存，因为课程项目的核心任务是自动物相识别。

## 5. 物理一致性设计

模型没有把 Bragg loss 强制加入主训练目标，而是采用更稳定的物理可靠性评价：

1. 模型预测晶格参数。
2. 根据 `hkl` 和预测晶格参数计算理论 2θ。
3. 与数据集中保存的参考峰位比较。
4. 输出 Bragg 峰位平均绝对误差。

该指标可以量化模型预测是否满足基本晶体学约束，属于 physics-informed reliability metric。

## 6. 输出结果

训练脚本：`src/train_multitask_cnn.py`

主要输出：

| 输出 | 用途 |
| --- | --- |
| `best_multitask_cnn.pt` | 最佳模型 |
| `metrics_summary.json` | 汇总指标 |
| `prediction CSV` | 单样本预测结果 |
| `confusion matrix` | 错误类别分析 |
| `training curves` | 训练过程可视化 |

评价指标包括物相准确率、晶系准确率、晶格 MAE 和 Bragg 峰位误差。

## 7. 运行示例

快速测试：

```powershell
python src\train_multitask_cnn.py `
  --epochs 1 `
  --batch-size 64 `
  --max-train-samples 512 `
  --max-val-samples 128 `
  --max-test-samples 128 `
  --device auto
```

正式训练示例：

```powershell
python src\train_multitask_cnn.py `
  --run-name residual_strong_lat003_ls005_v2_best_lattice `
  --epochs 50 `
  --batch-size 128 `
  --device auto `
  --label-smoothing 0.05 `
  --lattice-mode phase_residual `
  --residual-scale 0.02
```

## 8. 小结

单相模型不是简单黑箱分类器，而是结合物相识别、晶系识别、晶格回归和 Bragg 定律误差评价的多任务 XRD 模型。它为后续混相识别模型提供预训练基础，也为项目中的物理一致性与误差量化提供支撑。
