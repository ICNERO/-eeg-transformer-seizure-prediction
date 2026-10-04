# 整理记录与修改验证

本次整理创建公开归档副本，原始顶层研究文件保留。所有笔记本的逐文件来源、原始 SHA256、修改位置、方法、问题与验证结果记录在 [archive-manifest.json](archive-manifest.json)。

| 修改位置 | 修改方法与目的 | 对应验证 |
| --- | --- | --- |
| `notebooks/preprocessing/` 的 10 份笔记本 | 按早期步骤改用英文文件名；新增用途/来源说明，清除输出、执行状态、附件与本机路径 | nbformat、AST、输出和路径检查通过 |
| `notebooks/workflows/` 的 5 份笔记本 | 分组保存患者、对照及只处理发作记录的版本；保留算法参数，清理公开副本 | 同上 |
| `notebooks/models/` 的 2 份笔记本 | 保留新旧版本关系与模型代码，新增归档状态说明 | 同上；较新模型另做合成前向检查 |
| `notebooks/results/read_prediction_npy.ipynb` | 清理旧预测输出，使用相对结果目录，保留历史笔记本入口 | nbformat 与 AST 通过；未执行完整结果分析 |
| `notebooks/preprocessing/lowpass_downsampling_optional.ipynb` 原代码单元 0 行 657–658 | 合并残缺/重复 `elif`，修正该分支缩进，解决确定的语法错误 | 完整单元 AST 通过；缺原数据，未执行该分支 |
| MIT/Bonn 自动 FIF 清理调用 | 增加 `EEG_ENABLE_CLEANUP=1` 条件，默认保留中间文件；Bonn 同时尊重 `--keep_fif` | AST 通过；静态检查调用均由显式条件控制 |
| `scripts/eeg_model.py` | 从较新模型笔记本导出，保留训练/评估/预测 CLI | AST、`--help`、合成 loader/collate/模型检查通过 |
| `scripts/preprocess_mit.py`、`preprocess_bonn.py`、`preprocess_controls.py` | 从对应完整流程导出，保留既有 CLI 和信号处理实现 | AST 与三个 `--help` 入口通过；真实处理未执行 |
| `scripts/analyze_predictions.py` | 导出结果工具，用 `--output-dir` 替代固定目录，并同步更新控制台提示 | AST 与 `--help` 通过；完整结果分析仍需已有预测输出 |
| `README.md`、`docs/DATA.md`、`docs/KNOWN_LIMITATIONS.md`、`docs/VERIFICATION.md`、本文件 | 补齐竞赛背景、导航、运行命令、输入契约、数据缺失状态和已有实现限制 | 对照源代码审查，内部链接检查 |
| `requirements.txt`、`requirements-optional.txt`、`requirements-dev.txt` | 按源码导入重建基础/可选/验证依赖；记录历史版本锁定缺失 | 与导入清单及 Picard ICA 调用核对；未宣称完整环境复现 |
| `.gitignore` | 排除数据、权重、生成结果、凭证和本机验证环境 | 提交前核对发布文件清单 |
| `tools/verify_archive.py` | 对 18 份笔记本和 5 个脚本验证结构、语法、旧输出与本机路径 | 全量运行：0 failures |
| `tools/smoke_model.py` | 用临时合成特征检查真实导出模型的 loader、collate 和前向契约 | 输出 `(2,1)`，全部有限；原数据与性能未涉及 |
| `.github/workflows/archive-check.yml` | Python 3.12、只读内容权限的自动归档检查 | 命令在本机通过；远端运行结果以 Actions 为准 |

未对已知模型/预处理研究问题进行算法重设计；相关行为与后续复现条件见 [已知限制](KNOWN_LIMITATIONS.md)。没有重建或替代原测试集，没有为项目编造性能数字。
