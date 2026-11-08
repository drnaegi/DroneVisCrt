# 单图与双图的 ZeroNVS NeRF 对照实验

本实验使用 AerialMegaDepth 官方仓库同一场景的真实照片，比较单张参考图与两张不同角度参考图对 NeRF 的影响。它是本机适配的多参考图扩展，**不是 ZeroNVS 官方多图模式，也不是论文配置的严格复现**。

**已完成两组各 2000 步训练、各 120 帧导出和产物验证。本次双图输入改善了新增俯视训练图的重建，但没有得到可用的新视角几何，尚不适合高质量样本扩增。**

## 图片与实验划分

场景：`t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10`。

| 图片 | 内容 | 单图组 | 双图组 |
|---|---|---|---|
| `00572` | 原有地面参考图 | 训练 | 训练 |
| `00563` | 俯视照片，相对估计旋转约 89° | 不参与损失 | 训练 |
| `00616` | 另一地面角度，相对估计旋转约 22° | 评估 | 评估 |

三张照片均用于 MASt3R 的相机和全局几何估计。因此，第三张图虽然没有进入 NeRF 的 RGB、预测深度或 SDS 参考损失，**仍然通过相机估计间接提供了信息**。这里的评估是共享相机估计条件下的留出渲染检查，不能称为完全未见图像的泛化基准。

图片先经 EXIF 校正，用 `ImageOps.fit` 中心裁成 512×512，再以 LANCZOS 缩放为 256×256。来源 URL、原图 SHA256 与预处理记录位于 `inputs/multiview_a10/provenance.json`。预处理脚本会验证已有输入的像素，防止覆盖训练依据。

## 方法与公平性

两组重新训练，均使用同一套估计相机、相同随机种子、初始化方式、NeRF 结构、优化器、损失权重、目标相机分布及总射线预算；变化项是允许使用的真实训练参考图集合。

| 设置 | 两组共同配置 |
|---|---|
| 训练步数 | 每组 2000 |
| 总 RGB 射线 | 每步 4096；单图组每张 4096，双图组每张 2048 |
| NeRF | 16 层哈希网格，PyTorch CUDA 体渲染，空间收缩，独立密度/颜色 MLP |
| 每条射线采样 | 96 个区间；前半逆深度覆盖 0.15–2，后半线性覆盖 2–12 |
| 优化器 | Adam，学习率 0.005，betas=(0.9, 0.99) |
| 损失权重 | RGB=1500，预测深度=20，SDS=0.1 |
| 深度监督 | MASt3R 预测深度，置信度 >3；不是真值深度 |
| SDS | 预热 100 步，之后每 2 步一次；每组 950 次 |
| SDS 渲染 | 64×64，再交给官方教师编码；低于此前单图实验分辨率 |
| 相机路径 | 两个真实训练相机之间的位置线性插值与旋转四元数插值 |
| 导出 | 每组 120 帧 256×256 PNG，并排比较视频 24 FPS |

SDS 使用 ZeroNVS 官方 EMA 权重与官方 guidance 类。双图组针对目标相机选择空间距离最近的训练参考图，切换对应的 CLIP/VAE 图像条件和预测深度尺度；单图组始终使用 `00572`。各向异性日程的角度参数采用参考相机与目标相机的相对旋转角。这些多图选择及相机路径策略属于本实验扩展。

相机由 AerialMegaDepth MASt3R 模型对三图的六个有向图像对推理，再做 400 次全局对齐得到，未经真值相机验证。模型置信度和对齐损失不能直接证明相机精度。

最初检查发现，纯逆深度采样对俯视相机距离约 7 的区域覆盖不足，故在正式对照前统一改为分段采样并重启两组。被中止的单图检查点保留于 `outputs/multiview_a10/aborted_inverse_sampling/`，不参与报告。此前 `outputs/aerialmegadepth_00572/` 的 4000 步实验也不作为此次公平对照。

## 模型来源

- AerialMegaDepth：[官方仓库](https://github.com/kvuong2711/aerial-megadepth)。
- MASt3R 权重：`kvuong2711/checkpoint-aerial-mast3r`，固定 revision `0cf76e1da5c9b2bb7fb9d17cce03e0e6582c034e`。
- MASt3R safetensors SHA256：`093c3b4b3ad8294673774aae106309d5f6a1a975976bae574075fb7c5f94a67d`，已核验。
- ZeroNVS EMA 推理权重 SHA256：`5e0798ce8c3a3185eed71d9b4297af1417faa0ad86e632103ffb259bea867d3d`。提取来源与原官方源码版本见 `README_ZeroNVS.md`、`source_provenance.json`。

## 运行与验证

沿用本目录现有 `.venv-zeronvs`、已下载的上游源码和模型配置。新增流程不修改 AirSim 的 `.venv`。

```powershell
# 权重不存在时下载；固定版本，下载后核验 SHA256
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\download_aerial_checkpoint.py

# 准备并核对图片，估计相机与预测深度
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\prepare_multiview.py

# 依次训练两组并导出；已有完整检查点可复用，部分检查点不会静默覆盖
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\run_multiview_experiment.py

# 仅重做渲染与评估，无需扩散教师
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\run_multiview_experiment.py --mode evaluate

# 核对训练记录、采样覆盖、检查点重载像素、相机与视频帧数
.\.venv-zeronvs\Scripts\python.exe -X utf8 scripts\verify_multiview.py
```

## 结果与产物

| 视角 | 单图 PSNR / SSIM | 双图 PSNR / SSIM | 解释 |
|---|---|---|---|
| 原地面图 `00572` | 33.44 dB / 0.942 | 29.83 dB / 0.873 | 两组都参与训练；双图组单张图的射线预算减半 |
| 俯视图 `00563` | 11.77 dB / 0.341 | 32.56 dB / 0.910 | 仅双图组参与训练，提升主要体现对新增输入的拟合 |
| 留出地面图 `00616` | 13.65 dB / 0.204 | 12.24 dB / 0.267 | 两组均不参与 NeRF 损失；相机估计使用了该图 |

留出视角的 PSNR 下降 **1.41 dB**，SSIM 上升 **0.064**，指标方向不一致。对照图中，双图组的留出视角主要被模糊的绿色区域遮挡；插值路径也存在大量灰色空白、拉伸纹理和不连续几何。SSIM 的小幅提升不能说明建筑结构正确。这里应记录为**双图训练拟合成功，但新视角质量未达到可用水平**。

目前能确认的是，单靠增加这张大跨度俯视图，在当前相机估计、NeRF 结构、采样与训练预算下不足以恢复一致几何。可能因素包括相机/深度误差、地面到俯视的有限重叠、短训练预算及稀疏体渲染；本实验没有分别消融这些因素，不能确定具体原因。

后续更有价值的实验是收集同一目标连续、重叠充分的多张照片，先检查相机和多视图几何，再加入生成先验；不建议直接把当前两端视角之间的渲染当作真实扩增图。

数值结果保存在 `outputs/multiview_a10/results.json`，验收结果保存在 `verification.json`。PSNR 使用 256×256 RGB 的整图 MSE；SSIM 使用 7×7 均匀窗口、样本协方差校正和 RGB 通道平均。训练图的拟合指标与留出图的指标分别报告。

产物验收为 `PASS`：两组各有连续 2000 步有限数值记录、950 次非零 SDS 图像梯度、120 帧 PNG，以及可严格加载的检查点。留出相机重载渲染与已导出 PNG 的最大像素差均为 0。往返路径每组有 61 个不同画面，其余为对称返回帧；视频共 120 帧。高置信预测深度的射线距离全部位于采样范围内。两组训练各用时约 308 秒和 318 秒，峰值 PyTorch 分配显存约 8.07 GiB。**产物验收通过不代表视觉质量通过。**

- `comparison.png`：每行一张真实照片；各列依次为真实图、单图组渲染、双图组渲染。
- `comparison_path.mp4`：左侧单图组，右侧双图组；两个训练相机之间往返插值，不代表任意 360° 环绕能力。
- `path_contact_sheet.png`：插值路径的采样帧。
- `single/last.pt`、`multi/last.pt`：两组 NeRF 与优化器检查点。
- `single/renders/`、`multi/renders/`：每组 120 张 PNG。
- `cameras.pt`、`cameras.json`、`estimated_geometry.ply`：估计相机、深度与点云。
- `transforms.json`：每帧 OpenGL camera-to-world 矩阵及以 512 分辨率表示的内参；使用 256 图像时内参缩放 0.5。

这是单个场景、单个随机种子的探索实验，不能用来证明多图输入在所有场景都更好。是否可用于样本扩增，还需要检查几何、遮挡、纹理以及下游任务收益。
