# DroneVisCrt

无人机巡检视觉研究工作区，围绕地面到航拍视角生成、空地几何重建、单图新视角合成和仿真数据采集开展实验，同时保存巡检图文生成与样本扩增研究方案、多模态智能体开题报告及修订产物。

仓库包含第三方源码快照、本地适配脚本、配置、实验记录和可视化结果。模型权重、原始数据和运行环境需单独准备；克隆仓库后可直接查阅已有结果。

## 当前实验状态

以下状态依据仓库中已保存的实验记录；AerialMegaDepth、ZeroNVS 和 AirSim 的实测日期为 2026-11-04。

| 方向 | 已完成内容 | 结论与边界 | 结果入口 |
| --- | --- | --- | --- |
| SkyDiffusion | 保存地面到航拍生成样例、投影与指标核验、固定配置下的诊断结果 | 当前使用 CVACT 权重测试 G2A-3/Disaster 的 8 对诊断样本；任务匹配权重、官方划分及部分协议尚未核实，不能视为论文基准复现 | [对齐审计](results/alignment_audit/)、[固定配置结果](results/skydiffusion_alignment_locked/)、[诊断配置](configs/skydiffusion_disaster_diagnostic.json) |
| AerialMegaDepth | Aerial-DUSt3R、Aerial-MASt3R 严格加载权重并完成 CPU 双向推理，保存点图、置信度和点云预览 | 验证了几何预测链路，尚未验证论文重建精度；未执行全局对齐优化 | [CPU 实测报告](results/aerialmd_cpu/README.md) |
| ZeroNVS / MegaScenes | 使用 MegaScenes 发布的 ZeroNVS 微调权重，完成 -10°、+10° 两个新视角的 CPU 生成 | 输出为 256×256 RGB；相机参数为演示假设，没有目标视角真值，不报告重建精度；不含 NeRF/SDS 蒸馏 | [CPU 复现报告](results/zeronvs_cpu/README.md) |
| AirSim | Linux Blocks 场景启动、RPC、场景查询、位姿控制和相机信息检查通过 | 当前 CPU 软件渲染环境取图时崩溃，尚未成功采集 RGB、深度或分割图；GPU 渲染待验证 | [安装与运行检查](results/airsim_smoke/README.md) |
| Windows / CUDA 样本扩增 | RTX 5070 Ti 上完成单图 ZeroNVS NeRF 蒸馏，以及单图/双图对照训练与产物核验 | 新视角仍有模糊和几何失真，尚不适合高质量扩增；属于本地适配及多参考图扩展 | [模块说明](sample_augmentation/README.md)、[单图/双图对照](sample_augmentation/README_MultiView.md) |

AerialMegaDepth 的几何预测与 ZeroNVS 的 RGB 生成属于不同任务，不能直接作为同一种新视角生成方法比较。MegaScenes 代码目前用于 ZeroNVS 微调模型推理，尚未验证完整的 warp-plus-pose 流程。

## 内容导航

| 位置 | 内容 |
| --- | --- |
| [code/skydiffusion/](code/skydiffusion/) | SkyDiffusion 源码及本地推理修复 |
| [sample_augmentation/](sample_augmentation/) | Windows / CUDA ZeroNVS 单图与双图 NeRF 蒸馏、AirSim 下载工具及正式实验记录 |
| [code/aerial-megadepth/](code/aerial-megadepth/) | AerialMegaDepth 推理所需的部分源码、MASt3R/DUSt3R 依赖及示例图片 |
| [code/megascenes-nvs/](code/megascenes-nvs/) | MegaScenes 新视角合成源码快照及 ZeroNVS 微调配置 |
| [code/zeronvs-diffusion/](code/zeronvs-diffusion/) | 原版 ZeroNVS 扩散代码快照，供来源核对与参考 |
| [configs/](configs/) | SkyDiffusion 固定诊断配置、AirSim ComputerVision 场景配置 |
| [tools/](tools/) | 数据审计、权重下载、推理、评价、CPU 适配与仿真检查脚本 |
| [results/](results/) | 实验结果、运行日志、环境依赖记录、来源清单和核验材料 |
| [outputs/inspection_research_plan_20261015/](outputs/inspection_research_plan_20261015/) | 图文生成与扩增实验方案、实验矩阵及调研材料 |
| [outputs/proposal_revision_20261017/](outputs/proposal_revision_20261017/) | 开题报告修订稿、流程图和修订脚本 |
| [papers/](papers/)、[无人机航拍视角/](无人机航拍视角/) | 论文与中文阅读、调研材料 |
| 根目录中的 Markdown 和 PDF | 研究方案、投稿方向调研、论文清单及开题报告 |

研究文档入口：[完整实验方案](无人机巡检图文生成与扩增_完整实验方案.md)、[论文清单与复现调研](论文清单与复现调研.md)、[投稿方向建议](地面到无人机视角生成_投稿方向建议.md)。

## 已保存的结果

### AerialMegaDepth 几何预测

在 Intel Xeon Silver 4314、CPU FP32、16 个 PyTorch 线程下，官方示例图像对的后续三次双向推理平均耗时为：Aerial-DUSt3R **6.54 秒**，Aerial-MASt3R **6.15 秒**；进程峰值 RSS 分别为 **4.65 GiB** 和 **5.53 GiB**。推理计时不含模型加载、图像预处理、特征匹配、位姿估计、全局对齐和结果写盘。

查看 [几何预测对比](results/aerialmd_cpu/geometry_comparison.png)、[置信度对比](results/aerialmd_cpu/confidence_comparison.png) 和 [点云预览](results/aerialmd_cpu/pointcloud_comparison.png)。可视化按模型分别归一化，不能据此比较米制深度或重建准确率。

### ZeroNVS 新视角生成

使用 MegaScenes 的 ZeroNVS 微调模型 iteration 90000，CPU FP32 推理生成 -10° 和 +10° 视角，耗时分别为 **51.06 秒**和 **51.42 秒**。计时包含去噪与 VAE 解码，不含模型加载与参考图编码。DDIM 配置为 50 步，官方实现实际执行 49 次更新。

查看 [输入与生成结果对比](results/zeronvs_cpu/comparison.png) 和 [运行汇总](results/zeronvs_cpu/summary.json)。生成结果存在建筑与环境细节变化，不能当作目标视角的精确重建。

## 运行入口

### Windows / CUDA 样本扩增

该模块使用独立的 Windows 环境，命令在 `sample_augmentation/` 内执行。原实验环境为 Python 3.13.5、PyTorch 2.9.1+cu130 和 RTX 5070 Ti；安装、权重准备及 PowerShell 命令见 [模块说明](sample_augmentation/README.md)、[单图运行说明](sample_augmentation/README_ZeroNVS.md) 和 [单图/双图对照](sample_augmentation/README_MultiView.md)。

单图正式训练完成 4000 步，对照实验两组各完成 2000 步，产物核验见 [验证记录](sample_augmentation/VERIFICATION_ZeroNVS.md) 及模块内输出目录。参考视角拟合指标不代表新视角精度；对照实验的留出图参与了共享相机估计，不属于完全未见图像的泛化基准。训练检查点和模型权重未入库，克隆后需重新训练或另行提供检查点；复测前请使用新的输出目录，或备份并移走已有正式实验记录。

### Linux 实验环境约定

以下命令从仓库根目录的 **Linux/Bash** 环境执行，假定对应的独立 Python 环境和本地权重已准备好。`envs/*/bin/python` 是已有实验环境的位置，不会随 Git 克隆获得；迁移后可替换为实际解释器路径。每次复测应使用新的输出目录，保留已有结果。

### AerialMegaDepth CPU 推理

权重分别放在 `checkpoints/aerial-megadepth/checkpoint-aerial-dust3r/` 和 `checkpoints/aerial-megadepth/checkpoint-aerial-mast3r/`。来源和环境说明见 [实测报告](results/aerialmd_cpu/README.md) 与 [上游 README](code/aerial-megadepth/README.md)。

```bash
envs/aerialmd-cpu/bin/python tools/benchmark_aerialmd_cpu.py --model dust3r --threads 16 --runs 3 --output results/aerialmd_retest
envs/aerialmd-cpu/bin/python tools/benchmark_aerialmd_cpu.py --model mast3r --threads 16 --runs 3 --output results/aerialmd_retest
```

默认使用仓库中的官方示例对，可通过 `--images /absolute/path/image1.jpg /absolute/path/image2.jpg` 指定两张图片。[可视化脚本](tools/visualize_aerialmd_outputs.py) 当前读取固定的 `results/aerialmd_cpu/` 目录；处理其他输出目录前需调整路径。

### ZeroNVS CPU 推理

默认读取 `checkpoints/zeronvs-megascenes/model-inference.safetensors`。权重下载与非 EMA 推理参数提取说明见 [复现报告](results/zeronvs_cpu/README.md)，对应工具为 [下载器](tools/download_zeronvs_ms.py) 和 [参数提取脚本](tools/extract_zeronvs_inference.py)。

```bash
envs/zeronvs-cpu/bin/python tools/run_zeronvs_cpu.py --steps 50 --threads 16 --angles -10 10 --output results/zeronvs_retest
```

默认使用 AerialMegaDepth 示例中的地面建筑图片，可通过 `--image /absolute/path/input.jpg` 更换输入。

### AirSim 场景与取图检查

启动器依赖已有的 `/opt/airsim/Blocks-v1.8.1` 安装、`airsim` 服务账号、Xvfb 及 Vulkan 环境，需要 root 调用。安装细节见 [运行检查报告](results/airsim_smoke/README.md)。以下默认软件渲染命令用于复测当前故障：

```bash
bash tools/run_airsim_blocks.sh
```

在另一个终端运行检查：

```bash
envs/airsim/bin/python tools/smoke_test_airsim.py --output results/airsim_retest
```

具备可用 GPU、图形驱动和 Vulkan ICD 时，可用 `AIRSIM_RENDERER=gpu bash tools/run_airsim_blocks.sh` 启动 GPU 分支；该分支尚未验证。仅安装 CUDA 不满足场景渲染条件。

### SkyDiffusion 诊断与核验

数据准备、推理和评价入口分别为 [prepare_disaster_subset.py](tools/prepare_disaster_subset.py)、[run_skydiffusion.py](tools/run_skydiffusion.py) 和 [evaluate_skydiffusion.py](tools/evaluate_skydiffusion.py)。准备好诊断配置要求的权重、数据与清单后，可先使用 Python 运行预检：

```bash
python tools/skydiffusion_preflight.py --profile configs/skydiffusion_disaster_diagnostic.json --mode diagnostic
```

预检会检查权重及代码哈希、输入清单和配置。`diagnostic` 通过只表示诊断条件满足；当前资料不足以通过正式 `benchmark` 模式。详细证据见 [对齐状态](results/alignment_audit/alignment_status.json)。

## 数据、权重与环境

原始数据集、模型权重、数据压缩包和本机运行环境不纳入普通 Git 版本管理；本地文件保留原位。同步范围见 [GITHUB_UPLOAD_PLAN.md](GITHUB_UPLOAD_PLAN.md)，此前整理的大文件清单见 [workspace_large_files.csv](workspace_large_files.csv)；实际忽略规则以 [.gitignore](.gitignore) 为准。

| 内容 | 本地位置或依赖记录 |
| --- | --- |
| 原始数据 | `datasets/`，需按对应上游说明另行准备 |
| 模型权重 | `checkpoints/`；SkyDiffusion 下载与校验见 [下载脚本](tools/download_skydiffusion_checkpoint.py) |
| 独立运行环境 | `envs/`；根据各实验报告重建，避免混用模型依赖 |
| SkyDiffusion 依赖 | [requirements](tools/skydiffusion-requirements.txt)、[环境记录](tools/skydiffusion-environment.freeze.txt) |
| AerialMegaDepth 依赖 | [环境记录](results/aerialmd_cpu/environment.freeze.txt) |
| ZeroNVS 依赖 | [环境记录](results/zeronvs_cpu/environment.freeze.txt) |
| AirSim 依赖 | [requirements](tools/airsim-requirements.txt)、[已安装包记录](results/airsim_smoke/python_packages.txt) |

`results/` 下的 CPU 和 AirSim 测试在 Linux 容器中完成；Windows / CUDA 实验位于 `sample_augmentation/`，依赖与运行记录独立维护。部分脚本、数据清单和历史记录包含本机绝对路径；迁移后需按实际文件位置调整。环境冻结文件用于记录实验依赖，其中可能包含基础环境继承的包，不代表跨平台通用安装清单。

## 来源与许可

第三方源码、模型、数据和论文分别遵循各自的许可及使用条件，请查阅对应目录中的 `LICENSE`、`NOTICE` 和上游说明。`code/` 下的 AerialMegaDepth、MegaScenes 与 ZeroNVS 为 CPU 推理获取的部分源码快照，未包含所有上游训练、数据或演示资源；已获取文件的固定提交及校验记录见对应结果目录中的 `source_manifest.json`。Windows 样本扩增模块的来源与导入清单见 [source_provenance.json](sample_augmentation/source_provenance.json) 和 [IMPORT_MANIFEST.json](sample_augmentation/IMPORT_MANIFEST.json)。

研究方案中的计划与已完成实验应结合各结果目录的 README、原始日志和核验记录阅读，不能仅凭运行成功认定复现了论文指标。
