# DroneVisCrt

无人机巡检视觉研究工作区，包含地面到航拍视角生成的复现材料、巡检图文生成与样本扩增研究方案，以及多模态智能体开题报告及修订产物。

## 内容导航

| 位置 | 内容 |
| --- | --- |
| `code/skydiffusion/` | 第三方研究代码及本地复现修改；许可见该目录内的 `LICENSE` |
| `sample_augmentation/` | Windows / CUDA ZeroNVS 单图与双图 NeRF 蒸馏、AirSim 下载工具及正式实验记录；新视角质量仍未达到高质量扩增要求 |
| `configs/` | 实验配置 |
| `tools/` | 数据审计、模型下载、推理、评价与对齐检查脚本 |
| `results/` | 已保存的实验结果、日志、调研来源和核验记录 |
| `outputs/inspection_research_plan_20261015/` | 图文生成与扩增实验方案、实验矩阵及调研材料 |
| `outputs/proposal_revision_20261017/` | 开题报告修订稿、流程图和修订脚本 |
| `papers/`、`无人机航拍视角/` | 论文与调研参考材料 |
| 根目录中的 Markdown 和 PDF | 复现记录、研究方案、文献调研及原始开题报告 |

## 数据、权重与环境

原始数据集、模型权重、数据压缩包和本机运行环境不纳入普通 Git 版本管理；本地文件保留原位。具体清单见 `GITHUB_UPLOAD_PLAN.md` 及 `workspace_large_files.csv`。

- 数据准备方式见 `code/skydiffusion/README.md` 和 `tools/` 中的数据准备脚本。
- 模型下载及校验逻辑见 `tools/download_skydiffusion_checkpoint.py`。
- 环境依赖参考 `tools/skydiffusion-requirements.txt`、`tools/skydiffusion-environment.freeze.txt` 和上游环境说明。
- 部分实验脚本使用 `/root/DroneVisCrt` 下的本机绝对路径，迁移到其他机器后需调整配置和路径。

本仓库包含研究计划与不同阶段的复现记录，具体结论以对应的核验报告和原始结果为准。第三方代码、数据和论文分别遵循其原有许可及使用条件。
