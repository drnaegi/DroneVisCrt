# AerialMegaDepth CPU 推理实测

测试日期：2026-11-04（Asia/Shanghai）。两个模型均已完成严格权重加载和真实图像推理。

## 环境与输入

- CPU：Intel Xeon Silver 4314 @ 2.40 GHz；宿主机可见 64 个逻辑 CPU，容器 CPU 配额相当于 16 核。
- 容器内存上限：64 GiB；无可用 CUDA 设备。
- Python 3.8.10，PyTorch 2.0.0+cu118，CPU FP32；16 个算子线程，1 个 inter-op 线程。
- 虚拟环境：`/root/DroneVisCrt/envs/aerialmd-cpu`，使用 `--system-site-packages` 复用基础环境的 PyTorch；额外安装 einops 0.8.1、roma 1.8.0、trimesh 4.12.2，未更改基础环境的软件包。
- 使用官方 siteACC0003-finearts 示例对；按官方预处理缩放长边到 512，输入分别为 384×512 与 512×384（宽×高）。
- 使用官方 `inference`，batch size 为 1，包含两个方向 `(B,A)` 和 `(A,B)`，顺序执行两个模型。
- 每个模型首次推理作为预热，其后测量 3 次。没有编译 CUDA RoPE 扩展，使用官方 PyTorch 实现。

## 实测结果

| 模型 | 模型加载 | 首次双向推理 | 后续三次双向推理 | 平均耗时 | 进程峰值 RSS |
| --- | ---: | ---: | --- | ---: | ---: |
| Aerial-DUSt3R | 8.06 s | 9.09 s | 7.08 / 6.36 / 6.17 s | 6.54 s | 4.65 GiB |
| Aerial-MASt3R | 8.69 s | 8.98 s | 6.75 / 5.48 / 6.21 s | 6.15 s | 5.53 GiB |

推理耗时不含模型加载、图像预处理、结果写盘、特征匹配、位姿估计或全局对齐优化。
峰值 RSS 是整个进程的峰值，包含权重加载，不是纯推理阶段的内存增量。
这些结果只代表当前机器、这组输入和线程配置；两者约 0.4 秒的均值差距不足以证明普遍的速度优势。

两个模型均使用 `from_pretrained(..., local_files_only=True, strict=True)` 加载本地权重。
所有返回的点图、置信度，以及 MASt3R 描述子和描述子置信度均检查为有限数值。
这证明运行链路可用，不代表已验证论文中的重建精度。

## 复现命令

```bash
cd /root/DroneVisCrt
envs/aerialmd-cpu/bin/python tools/benchmark_aerialmd_cpu.py --model dust3r
envs/aerialmd-cpu/bin/python tools/benchmark_aerialmd_cpu.py --model mast3r
```

使用自己的图片并保留本次基准结果：

```bash
/root/DroneVisCrt/envs/aerialmd-cpu/bin/python \
  /root/DroneVisCrt/tools/benchmark_aerialmd_cpu.py \
  --model mast3r --threads 16 --runs 3 \
  --images /absolute/path/image1.jpg /absolute/path/image2.jpg \
  --output /root/DroneVisCrt/results/aerialmd_cpu_custom
```

## 文件与来源

- `dust3r_benchmark.json`、`mast3r_benchmark.json`：逐次耗时、输入尺寸、输出结构、数值检查和内存统计。
- `dust3r_geometry.npz`、`mast3r_geometry.npz`：第一个方向 `(B,A)` 的两个视图点图和置信度；view1 对应输入列表中第二张图，view2 对应第一张图，两者点图都在 view1 坐标系中。
- `dust3r_run.log`、`mast3r_run.log`：完整运行输出。
- `environment.freeze.txt`：虚拟环境中可见的软件包版本，包括从基础环境继承的依赖。
- `source_manifest.json`：实际下载文件及其 Git blob SHA-1 校验值。
- `geometry_comparison.png`：输入图片与两个模型的参考相机 Z 坐标图；两行都使用相机 B 的坐标系，每个模型独立进行 2–98% 分位数显示归一化，不表示米制深度。
- `confidence_comparison.png`：模型置信度可视化，按模型独立归一化后的对数分数，不是准确率概率。
- `pointcloud_comparison.png`：两个视图预测点图的彩色点云预览；颜色取自输入，显示时每 4 像素采样、保留每个视图较高的 50% 置信度并裁掉坐标极端值；没有进行全局对齐优化。

可视化复现：`/root/DroneVisCrt/envs/aerialmd-cpu/bin/python /root/DroneVisCrt/tools/visualize_aerialmd_outputs.py`。

官方源码仓库：https://github.com/kvuong2711/aerial-megadepth

固定提交：`cdf478ff4fdaf888fb91c940bc48d55095e2cdbd`。

完整 Git 克隆因连接速度过慢而停止，改为获取 CPU 推理所需源码和示例；多数文件通过 jsDelivr 获取，示例图片之一通过 GitHub API 获取，所有保留的源文件和图片均与 GitHub 树中的 blob 哈希核对。
本地 `/root/DroneVisCrt/code/aerial-megadepth` 是用于本次推理的部分源码快照，没有 Git 元数据，部分训练、数据生成和说明文件未下载，不能视作完整训练仓库。
