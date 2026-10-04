# 本次验证

本页对应公开归档整理，不是原竞赛实验结果。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| Notebook 格式 | 18 / 18 通过 | `nbformat.validate`，首单元归档说明 |
| Notebook 源码 | 18 / 18 通过 | 完整 Python AST 语法；一个历史残缺 `elif` 已最小修复 |
| 脚本语法 | 5 / 5 通过 | Python AST 语法检查 |
| 输出与路径清理 | 通过 | 公开代码无旧输出、执行计数和 Windows 本机绝对路径 |
| 五个脚本帮助入口 | 5 / 5 通过 | 各脚本 `--help` 均正常退出 |
| 合成结构检查 | 通过 | 两个临时 `(1,8,40,128)` 数组，经 loader/collate 得 `(2,8,40,16)`，模型输出 `(2,1)` 且全部有限 |
| 真实数据流程与性能 | 未执行 | 缺少原 EEG、原测试划分和训练权重 |

本机使用项目内隔离的 nbformat、MNE、pykalman、tabulate 等验证依赖，并将 MNE 配置目录放在项目内；它们没有纳入仓库。帮助入口通过不代表真实信号处理已经验证。原笔记本环境元数据是 Python 3.12.3，完整历史环境未恢复。

## 重复本次检查

```bash
python -m pip install nbformat
python tools/verify_archive.py
```

加载器与模型的合成结构检查需要 `requirements.txt` 中的模型基础依赖：

```bash
python tools/smoke_model.py
```

该脚本固定合成输入的随机种子，使用 CPU、短序列及较小隐藏维度。它不训练模型、不读取原始数据、不保存权重，也不计算准确率。合成特征临时写在仓库内 `.local/`，结束时删除。

GitHub Actions 工作流只执行归档结构检查；它不训练、不执行 EEG 预处理，也不自动运行模型结构检查。
