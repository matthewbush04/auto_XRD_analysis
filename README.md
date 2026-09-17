# XRD 自动化材料表征项目

本项目实现了一个面向电池材料 XRD 数据的 AI 辅助自动表征 workflow。项目核心不是单独做一个分类模型，而是把材料数据库、合成 XRD 数据构建、单相识别、混相识别、误差量化、物理一致性检查和可视化结果整合成一条可嵌入高通量表征平台的算法链路。

当前最终主线包括：

1. 单相 XRD multi-task CNN：识别物相、晶系，并回归晶格参数。
2. 混相 XRD multi-label CNN：识别主相、副相、杂相或分解产物共存的多相样品。
3. Bragg-law 物理一致性评估：用预测晶格参数反算峰位误差，量化模型可靠性。
4. 本地可视化 dashboard：用于展示模型对比、混相识别结果和关键性能指标。

主动学习实验是项目的扩展模块之一，目前文件仍在完善中。当前 README 的主体先围绕已经闭环的单相模型、混相 v1/v2/v3 数据与 dashboard 展开；主动学习部分建议在最终展示中作为 ongoing extension 或 future workflow loop 说明。

## 1. 项目目标

课程要求是构建一个基于文献和开放材料数据库的 AI-powered autonomous material characterization workflow。本项目选择 X-ray diffraction, XRD 作为表征场景，面向电池材料体系完成以下目标：

- 基于 Materials Project 结构数据构建可追溯材料数据库。
- 使用 pymatgen 生成合成 XRD pattern，并加入峰宽、峰位漂移、背景、噪声和强度扰动。
- 训练单相模型，实现 phase identification、crystal-system classification 和 lattice-parameter regression。
- 训练混相模型，实现多标签 phase detection，覆盖主相 + 副相 / 杂相 / 分解产物共存场景。
- 输出 accuracy、F1、minor phase recall、confusion matrix、lattice MAE、Bragg peak MAE 等误差指标。
- 通过 dashboard 提供结果可视化界面，便于现场展示和后续集成。

## 2. 高通量表征 workflow 设计

本项目中的 AI 模块可以嵌入如下高通量 XRD 平台：

```text
批量样品制备
  -> 自动进样 / 样品盘编号
  -> XRD 自动扫描与文件保存
  -> XRD pattern 预处理
  -> 单相 / 混相模型自动推理
  -> 误差量化与物理一致性检查
  -> dashboard 展示与报告生成
  -> 人工只复核低置信度或高风险样品
```

需要强调的是，本项目主要完成 AI 算法核心与结果展示层。硬件平台部分以工程设计方案形式呈现，后续可以与自动进样器、实验室信息管理系统或仪器数据目录监听程序集成。

## 3. 效率提升说明

传统 XRD 分析通常需要人工完成 peak matching、phase search、晶系判断和报告整理。对于批量样品，人工分析时间常常成为瓶颈。

本项目的效率提升主要发生在 analysis stage：

| 环节 | 传统流程 | AI workflow |
| --- | ---: | ---: |
| 单个 XRD pattern 物相判断 | 约 10-20 min/sample | 秒级推理 |
| 晶系与晶格参数初步分析 | 约 5-10 min/sample | 自动输出 |
| 批量结果汇总 | 人工整理 | 自动生成 CSV / 图表 / dashboard |
| 人工参与方式 | 每个样品都完整分析 | 重点复核低置信度样品 |

因此，在批量 XRD 数据已经采集完成的前提下，模型可以把平均人工分析时间从十几分钟降低到数分钟内复核，满足 50% 以上 mean analysis time reduction。对于大批量样品，模型可连续处理数百条 pattern，analysis throughput 可达到传统人工逐条分析的 10 倍以上。

该效率结论针对 XRD 数据解释和结果整理阶段，不声称仪器扫描本身速度提升 10 倍。

## 4. 项目结构

```text
Project/
  data/
    raw/cif/                         # Materials Project CIF 和结构文件
    metadata/                        # 材料索引、标签映射和摘要
    processed/
      single_phase/                  # 单相 train/val/test 数据
      mixed_phase/
        mixed_v1/                    # 随机混相 baseline
        mixed_v2_hard_eval/          # 独立 hard benchmark
        mixed_v3_hard_augmented/     # 最终推荐混相训练数据
        mixed_v5_battery_*/          # 主动学习扩展数据，仍在完善中
  src/
    download_materials.py            # Materials Project 数据下载
    generate_single_phase_dataset.py # 单相 XRD 数据生成
    generate_mixed_phase_dataset.py  # mixed_v1 数据生成
    generate_mixed_phase_hard_eval.py       # mixed_v2 hard benchmark
    generate_mixed_phase_hard_augmented.py  # mixed_v3 targeted augmentation
    model_multitask.py               # 单相 multi-task CNN
    train_multitask_cnn.py           # 单相训练和评估
    model_mixture.py                 # 混相 multi-label CNN
    train_mixed_phase_cnn.py         # 混相训练和评估
    evaluate_mixed_phase_cnn.py      # 独立 hard benchmark 评估
    physics.py                       # Bragg-law 一致性指标
    active_learning_*.py 等          # 主动学习扩展脚本，当前不是主线闭环
  models/                            # 训练得到的模型 checkpoint
  results/                           # 指标、预测文件、混淆矩阵和训练曲线
  logs/                              # 训练日志
  docs/                              # 数据集、模型和混相实验报告
  figures/                           # 报告和 dashboard 使用的图片
  model_comparison_visualization/    # 单相模型对比 dashboard
  mixed_phase_dashboard/             # 混相识别 dashboard
  check_environment.py               # 环境与数据完整性检查
  requirements.txt
  README.md
```

## 5. 数据集

### 5.1 材料来源

材料结构来自 Materials Project。当前项目是 battery-focused pilot，覆盖 cathode、anode、solid electrolyte 和 reference/decomposition product 等电池相关类别。

当前数据库概况：

```text
accepted material records: 211
phase labels:              207
ordered structures:        211
download failures:         0
```

材料记录可追溯到：

```text
data/metadata/materials_index.csv
data/metadata/materials_index.json
data/raw/cif/
```

### 5.2 单相数据

单相数据位于：

```text
data/processed/single_phase/
```

数据规模：

```text
phase labels:       207
material records:   211
total patterns:     43,470
XRD length:         3501 points
2theta range:       10-90 degrees
wavelength:         Cu K-alpha
```

split：

```text
train:       31,050
val:          4,140
test_normal:  4,140
test_hard:    4,140
```

### 5.3 混相数据

混相数据位于：

```text
data/processed/mixed_phase/
```

保留的最终主线数据包括：

```text
mixed_v1/
  random-mixture baseline，用于普通混相训练和对照。

mixed_v2_hard_eval/
  独立 hard benchmark，只用于评估，不参与训练。
  包括 low-fraction minor phase、high-overlap phase pairs 和 battery-relevant mixtures。

mixed_v3_hard_augmented/
  最终推荐混相训练数据。
  在 mixed_v1 基础上加入 targeted hard augmentation。
```

`mixed_v3_hard_augmented` 的规模：

```text
train:       43,050
val:          5,740
test_normal:  4,140
test_hard:    4,140
```

## 6. 模型设计

### 6.1 单相 multi-task CNN

输入是一条单相 XRD intensity curve，输出：

- phase label，207 类分类。
- crystal system，7 类分类。
- lattice parameters，预测 a, b, c。

核心设计：

- 1D-CNN backbone 提取 XRD 峰位和峰形特征。
- AdaptiveAvgPool1d(32) 保留粗粒度峰位信息。
- phase head、crystal-system head、lattice head 共享 backbone。
- phase-residual lattice prediction 利用 phase-specific mean lattice 作为先验。
- Bragg-law consistency metric 用于可靠性评估。

最终推荐单相 checkpoint：

```text
models/residual_strong_lat003_ls005_v2_best_lattice/best_multitask_cnn.pt
```

### 6.2 混相 multi-label CNN

输入是一条可能包含多个 phase 的 XRD curve，输出 207 个 phase 的存在概率。

核心设计：

- 使用 sigmoid + BCEWithLogitsLoss 进行 multi-label phase detection。
- 支持主相、副相、杂相和分解产物共存。
- 使用 single-phase checkpoint 作为 pretrained backbone。
- 使用 validation threshold calibration 控制 precision/recall trade-off。
- 使用 minor phase recall、top-k recall 和 false positives/sample 评估真实混相场景。

最终推荐混相 checkpoint：

```text
models/mixed_phase/mixed_phase_cnn_v3_hard_aug_presence_50ep/best_mixed_phase_cnn.pt
```

## 7. 最终结果摘要

### 7.1 单相模型

最终推荐单相模型在 normal test set 上表现稳定：

| 指标 | test_normal | test_hard |
| --- | ---: | ---: |
| phase accuracy | 99.879% | 97.729% |
| phase top-3 accuracy | 100.000% | 99.589% |
| crystal-system accuracy | 99.928% | 97.126% |
| lattice MAE mean | 0.045 A | 0.448 A |
| Bragg peak MAE | 0.312 deg | 2.132 deg |

normal split 更接近常规实验扰动；hard split 用更强噪声、背景、峰位偏移和峰宽扰动进行压力测试。

### 7.2 混相模型

最终推荐混相模型：

```text
mixed_phase_cnn_v3_hard_aug_presence_50ep
```

在普通混相测试集上：

| 指标 | test_normal | test_hard |
| --- | ---: | ---: |
| micro precision | 0.921 | 0.838 |
| micro recall | 0.895 | 0.761 |
| micro-F1 | 0.908 | 0.797 |
| minor phase recall | 0.821 | 0.606 |
| top-3 all phases hit | 0.868 | 0.678 |
| false positives/sample | 0.169 | 0.323 |

在独立 hard benchmark 上，相比 v1 random-mixture baseline，v3 targeted hard augmentation 对 battery-relevant mixture 有明显提升：

```text
micro-F1:           0.739 -> 0.802
minor phase recall: 0.403 -> 0.574
```

这说明 targeted hard augmentation 能改善真实电池样品中主相 + 少量副相 / 杂相 / 分解产物共存的识别能力。

## 8. 可视化与展示入口

### 8.1 单相模型对比 dashboard

```text
model_comparison_visualization/model_comparison_dashboard.html
```

该 dashboard 展示：

- 不同单相模型 run 的 phase accuracy。
- crystal-system accuracy。
- lattice MAE。
- Bragg-law consistency。
- normal / hard split 的性能差异。

重新生成命令：

```powershell
python model_comparison_visualization/generate_model_comparison_visuals.py
```

### 8.2 混相识别 dashboard

```text
mixed_phase_dashboard/mixed_phase_dashboard.html
```

该 dashboard 展示：

- mixed-phase recognition 为什么是 multi-label task。
- v1 random baseline、v2 hard diagnosis、v3 hard augmentation 的关系。
- micro-F1、minor recall、top3 all-hit、false positives/sample 等指标。
- battery-relevant mixture、低含量副相和峰重叠场景下的性能变化。

重新生成命令：

```powershell
python mixed_phase_dashboard/generate_mixed_phase_dashboard.py
```

后续如果制作整合版本地网站 dashboard，建议把这两个 dashboard 的核心内容合并为一个现场展示入口：

```text
项目目标 -> 高通量 workflow -> 数据集 -> 模型 -> 结果 -> case study -> 效率提升
```

## 9. 环境与运行

建议使用 Python 3.11。

安装依赖：

```powershell
pip install -r requirements.txt
```

环境和数据检查：

```powershell
python check_environment.py
```

如果使用 GPU，建议安装与本机 CUDA 匹配的 PyTorch 版本。

## 10. 复现实验命令

### 10.1 单相模型训练

```powershell
python -u src/train_multitask_cnn.py `
  --run-name residual_strong_lat003_ls005 `
  --epochs 30 `
  --batch-size 128 `
  --device auto `
  --label-smoothing 0.05 `
  --lambda-lat 0.03 `
  --strong-lattice-head `
  --lattice-mode phase_residual `
  --residual-scale 0.02
```

### 10.2 混相模型训练

```powershell
python src/train_mixed_phase_cnn.py `
  --data-dir data/processed/mixed_phase/mixed_v3_hard_augmented `
  --run-name mixed_phase_cnn_v3_hard_aug_presence_50ep `
  --epochs 50 `
  --batch-size 128 `
  --device auto `
  --pretrained-checkpoint models/residual_strong_lat003_ls005_v2_best_lattice/best_multitask_cnn.pt `
  --disable-fraction-level-head `
  --lambda-fraction-level 0 `
  --disable-weighted-sampling `
  --max-predictions 3
```

### 10.3 独立 hard benchmark 评估

```powershell
python src/evaluate_mixed_phase_cnn.py `
  --checkpoint models/mixed_phase/mixed_phase_cnn_v3_hard_aug_presence_50ep/best_mixed_phase_cnn.pt `
  --data-dir data/processed/mixed_phase/mixed_v2_hard_eval `
  --output-dir results/mixed_phase/hard_eval/v3_hard_aug_presence_on_mixed_v2_hard_eval `
  --threshold-mode all `
  --max-predictions 3 `
  --device auto `
  --batch-size 128 `
  --overwrite
```

## 11. 文档索引

```text
docs/dataset_construction_report.md
  单相数据集构建、Materials Project 下载策略、XRD 生成参数。

docs/cnn_model_report.md
  单相 multi-task CNN、loss、Bragg-law reliability metric、训练输出。

docs/mixed_phase_dataset_report.md
  mixed_v1 随机混相数据设计。

docs/mixed_phase_hard_eval_report.md
  mixed_v2 独立 hard benchmark 设计与 v1 baseline 诊断。

docs/mixed_phase_hard_augmented_report.md
  mixed_v3 targeted hard augmentation 数据设计。
```


## 12. 当前限制与后续方向

当前项目仍有以下限制：

1. 数据来自 Materials Project 计算结构和合成 XRD，不等价于真实实验 PXRD。
2. 当前数据库是 battery-focused pilot，尚未扩展到所有材料体系。
3. 高通量硬件平台目前是工程设计方案，不包含真实仪器控制代码。
4. 低含量 minor phase，尤其 5% 附近，仍是混相识别中的主要难点。
5. Bragg-law 模块主要用于 reliability evaluation，不是默认强物理 loss 训练。

建议最终展示时把后续方向收束为：

- 接入真实实验 XRD 数据进行 domain adaptation。
- 扩展材料范围和数据库规模。
- 将 dashboard 与自动进样、仪器采集目录和报告生成系统连接。
- 将主动学习作为下一阶段工作，用于自动选择最有价值的样品补充训练。
