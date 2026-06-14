# 混相困难增强训练集简要报告

## 1. 目标

`mixed_v3_hard_augmented` 是在困难评估之后构建的增强训练集。它保留 `mixed_v1` 的随机混合样本，并只在训练集和验证集中加入困难样本。

目标是在不明显损失普通混相识别能力的前提下，提高：

- 少量相召回率；
- 高峰重叠样本识别能力；
- 电池相关组合的鲁棒性。

## 2. 数据位置

```text
data/processed/mixed_phase/mixed_v3_hard_augmented
```

独立困难评估集仍然是：

```text
data/processed/mixed_phase/mixed_v2_hard_eval
```

注意：`mixed_v2_hard_eval` 只用于最终评估，不进入训练。

## 3. 文件规模

| 文件 | 样本数 | 说明 |
| --- | ---: | --- |
| `train.npz` | 43,050 | `mixed_v1/train` + 12,000 个困难样本 |
| `val.npz` | 5,740 | `mixed_v1/val` + 1,600 个困难样本 |
| `test_normal.npz` | 4,140 | 沿用普通测试集 |
| `test_hard.npz` | 4,140 | 沿用普通困难测试集 |
| `dataset_manifest.json` | - | 生成记录 |

## 4. 增强策略

| 划分 | 少量相 | 峰重叠 | 电池相关 | 新增总数 |
| --- | ---: | ---: | ---: | ---: |
| train | 6,000 | 4,200 | 1,800 | 12,000 |
| val | 800 | 560 | 240 | 1,600 |

比例设计：

- 50% 为低比例少量相；
- 35% 为高重叠物相对；
- 15% 为电池相关组合。

少量相比例包括 95/5、90/10、85/15、80/10/10 和 85/10/5。峰重叠样本来自余弦相似度最高的 10% 物相对。

## 5. 训练示例

该实验用于隔离“困难数据增强”的作用，因此建议使用不带比例辅助头的混相 CNN：

```powershell
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

训练后在独立困难评估集上测试：

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

## 6. 成功标准

与 v1 基线相比，增强训练应满足：

- `test_minor` 少量相召回率明显提升；
- 5% 和 10% 少量相识别改善；
- `test_overlap` 的 micro-F1 不明显下降；
- 普通 `test_normal` 的 micro-F1 下降不超过约 0.01-0.02。

参考基线：

| 模型 | 测试集 | micro-F1 | 少量相召回率 |
| --- | --- | ---: | ---: |
| v1 仅存在性 | 少量相 | 0.644 | 0.209 |
| v1 仅存在性 | 峰重叠 | 0.788 | 0.585 |
| v1 仅存在性 | 电池相关 | 0.739 | 0.403 |

## 7. 小结

困难增强训练集的作用是把模型在困难评估中暴露出的弱点重新加入训练分布，但保持测试集独立。这样可以证明性能提升来自更有针对性的数据构建，而不是测试集泄漏。
