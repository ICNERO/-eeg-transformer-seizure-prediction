# 数据来源与输入契约

## 原始数据状态

项目持有人确认原始测试数据已找不到。本仓库没有 EEG 原始记录、特征 NPY、固定测试划分、训练权重和人工真值。也没有用重新下载或合成的数据替代原竞赛测试数据。

## 公开数据来源

| 来源 | 官方入口 | 格式与本仓库的关系 |
| --- | --- | --- |
| CHB-MIT Scalp EEG Database v1.0.0 | [PhysioNet 数据页面](https://physionet.org/content/chbmit/1.0.0/) | 包含 EDF 和发作起止信息；`preprocess_mit.py` 解析 CHB 风格摘要。重新取得文件后仍需核对通道与参数 |
| Bonn EEG | [波恩大学医院 EEG 下载页面](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/) | 官方提供 A–E 组单通道 TXT，采样率 173.61 Hz；与历史 `preprocess_bonn.py` 的多通道 EDF/摘要接口不同 |
| 正常对照 / PNXX | 原文件仅保留相关目录标签 | 无法从现有文件确认具体数据来源、受试者关系和许可；需要重新提供来源与标签说明 |

CHB-MIT 的官方页面提供数据引用 DOI [10.13026/C2K01R](https://doi.org/10.13026/C2K01R) 和使用条件。Bonn 官方页面要求引用 Andrzejak 等人的 2001 年论文，Phys. Rev. E **64**, 061907。请按官方页面引用使用的数据，仓库没有重新分发这些数据。

“波恩”只是保留原工作流名称，其解析器处理 `File name`、`Registration start time`、`Seizure start/end time` 等 EDF 摘要字段，也尝试 CHB 风格格式。原始单通道 TXT 到此接口的转换步骤没有保留下来，不能据此宣称官方 Bonn 数据可直接运行。

## EDF 工作目录

```text
data/chb-mit/chb01/
├── chb01-summary.txt
├── chb01_01.edf
└── ...

data/controls/control01/
├── recording.edf
└── ...
```

这些目录是示例。脚本按输入工作目录写入 FIF、NPY 和图表，运行前应准备数据工作副本。患者流程按发作注释构造常规、发作前、发作中、发作后片段；对照流程每个 EDF 最多取开头 60 秒。各版本并不是参数完全相同的统一预处理器。

## 模型特征目录

```text
data/features/
├── patients/
│   └── patient01/
│       └── example_stockwell_selected8ch.npy
└── controls/
    └── control01/
        └── example_stockwell_selected8ch.npy
```

- 每个根目录下有受试者子目录；文件名包含 `stockwell`，选通道结果通常包含 `selected8ch` 或 `selected_8ch`。
- 集成预处理输出契约为 `(epochs, channels, frequencies, time)`，选通道后是 `(E, 8, 40, T)`。
- 较新 `EEGDataset` 对四维文件只取第一个 epoch，沿时间轴 `::8` 后截断或补零。训练、评估与预测入口固定 `max_length=240`，得到 `(B, 8, 40, 240)`。
- 标签由根目录确定：患者为 `1`，对照为 `0`。加载器没有读取预处理生成的 `*_labels.npy`。
- `deep_learning_model_revision.ipynb` 是较旧模型，多个 epoch 的轴处理存在历史问题；当前导出入口来自 `deep_learning_model.ipynb`。

请保证特征为适合模型读取的实数幅值数组，明确各通道对应的电极以及实际频率网格。形状兼容不等于患者与对照的物理含义一致；默认频率网格的不一致见 [已知限制](KNOWN_LIMITATIONS.md)。

## 结果文件

`analyze_predictions.py` 接受预测目录，读取 `predictions.npy`、`subject_predictions.npy` 和 `detailed_predictions.csv`，生成统计图与 Excel。其中 NPY 查看器使用 `allow_pickle=True`；只打开可信来源的结果文件。预测分布本身不包含正确率所需的人工真值。
