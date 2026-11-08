# 无人机视角样本扩增实验

本模块汇总单图 ZeroNVS NeRF 蒸馏、单图与双图对照实验，以及 AirSim Blocks 下载脚本。它是 Windows / RTX 5070 Ti 上的探索性适配实现。

**已有实验完成训练和产物核验，但新视角仍存在模糊与几何失真，尚不能直接作为高质量扩增样本。**

## 入口

- [单图实验说明](README_ZeroNVS.md)：方法、运行参数、模型来源与限制。
- [单图/双图对照](README_MultiView.md)：共享相机估计、训练划分、结果及公平性边界。
- [单图执行验证](VERIFICATION_ZeroNVS.md)：已有本机运行的验证记录。
- `scripts/`：训练、下载、渲染与验收脚本。
- `configs/`、`inputs/`：实验配置、输入图像及来源记录。
- `outputs/aerialmegadepth_00572/`、`outputs/multiview_a10/`：正式实验的图像、视频、数值与历史验证报告。
- `source_provenance.json`：上游版本与关键源码 SHA256；`environment_zeronvs.json`：原运行环境快照。

## 运行位置与依赖

命令均在本模块目录执行。先建立独立 Python 环境，并参考 `environment_zeronvs.json` 和各上游依赖说明安装依赖；环境快照不是可直接安装的 requirements 文件。原实验使用 Python 3.13.5、PyTorch 2.9.1+cu130 和 NVIDIA CUDA GPU。

`ZeroNVS/` 与 `third_party/aerial-megadepth/` 保留本地已有上游源码快照及许可文件，路径布局与脚本一致；它们不是本项目原创代码，也不作为 Git submodule 管理。使用和再分发应遵守上游许可及数据来源要求。

权重不随源码提交。按实验说明运行 `scripts/inspect_remote_checkpoint.py`、`scripts/download_inference_weights.py` 和 `scripts/download_aerial_checkpoint.py` 获取所需权重并核验。

## 导入范围

保留源码、配置、输入、来源记录及两组正式实验的可视化与数值记录。未包含虚拟环境、Python 缓存、模型权重、训练检查点、预测中间张量、点云、AirSim 运行包、下载缓存和中止实验。

`verification.json`、`status.json` 和验证文档记录的是原工作区已经执行的实验。由于检查点等大文件未入库，克隆后不能仅凭这些记录重新执行完整产物验收；需要重新训练或自行提供原检查点。训练前请使用新的输出目录，或备份并移走已归档的正式实验记录，避免与新运行混用。

`AirSim/download_blocks.py` 可重新下载模拟器运行包。完整文件清单及 SHA256 记录在 `IMPORT_MANIFEST.json`；本模块禁用 Git 换行转换，以保留来源记录与文件哈希。
