# 混相 XRD 数据集简要报告

## 1. 目标

混相数据集用于模拟真实样品中多种物相共存的 XRD 谱图，并训练多标签识别模型。模型需要判断样品中包含哪些物相，而不是只输出一个类别。

## 2. 数据来源

生成脚本：`src/generate_mixed_phase_dataset.py`

输入为单相数据集中的谱图和标签。混相样本由 1-3 个单相谱图按比例线性组合，并加入归一化处理。

数据目录：

```text
data/processed/mixed_phase/mixed_v1/
```

## 3. 数据规模

| 划分 | 样本数 | 谱图形状 | 1 相 | 2 相 | 3 相 |
| --- | ---: | --- | ---: | ---: | ---: |
| `train` | 31,050 | `[31050, 3501]` | 3,250 | 18,713 | 9,087 |
| `val` | 4,140 | `[4140, 3501]` | 429 | 2,526 | 1,185 |
| `test_normal` | 4,140 | `[4140, 3501]` | 410 | 2,511 | 1,219 |
| `test_hard` | 4,140 | `[4140, 3501]` | 428 | 2,460 | 1,252 |

总样本数为 43,470。该规模与单相数据集保持一致，便于横向比较。

## 4. 生成策略

主要设计：

- 保持 train、val、test 的划分隔离，避免数据泄漏。
- 混合相数限制为 1-3 相，先建立稳定基线。
- 二相和三相样本设置主相与少量相比例。
- 保存每个组分的物相编号、比例和主相信息。

第一版不生成 4 相或 5 相样本，因为相数过多会显著增加峰重叠和少量相识别难度，不利于建立稳定基线。

## 5. 保存字段

| 字段 | 含义 |
| --- | --- |
| `X` | 混相 XRD 谱图 |
| `y_multi` | 多标签物相向量 |
| `component_phase_indices` | 组成物相编号 |
| `component_fractions` | 组成比例 |
| `n_phases` | 物相数量 |
| `major_phase_index` | 主相编号 |
| `y_fraction_level` | 相含量等级标签 |

这些字段支持混相识别、主相识别、少量相召回率分析和 dashboard 展示。

## 6. 训练关系

训练脚本：`src/train_mixed_phase_cnn.py`

推荐命令：

```powershell
python src\train_mixed_phase_cnn.py `
  --data-dir data\processed\mixed_phase\mixed_v1 `
  --run-name mixed_phase_cnn_v1_pretrained `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --pretrained-checkpoint models\residual_strong_lat003_ls005_v2_best_lattice\best_multitask_cnn.pt
```

主要指标：

| 指标 | 含义 |
| --- | --- |
| micro-F1 | 整体多标签识别能力 |
| macro-F1 | 各物相平均识别能力 |
| subset accuracy | 整个物相集合完全预测正确的比例 |
| major phase accuracy | 主相识别准确率 |
| minor phase recall | 少量相召回率 |

## 7. 局限性

- `mixed_v1` 主要是随机混合基线，不专门针对困难场景。
- 少量相、强峰重叠和电池相关组合需要单独困难评估集。
- 混合策略尚未显式考虑真实实验中的取向效应、晶粒尺寸和仪器函数。

## 8. 小结

`mixed_v1` 将项目从单相识别扩展到更接近真实样品的混相识别。它的作用是建立稳定多标签基线，后续困难评估、困难增强训练和主动学习都以此为基础。
