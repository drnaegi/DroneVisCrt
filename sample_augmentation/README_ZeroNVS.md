# AerialMegaDepth 示例图的 ZeroNVS NeRF 蒸馏

本目录用于在 Windows / RTX 5070 Ti 16GB 上执行单图 → ZeroNVS SDS → NeRF → 多视角渲染的完整流程。

这是兼容本机的适配实现，**不是论文默认训练配置的严格复现**。真实执行进度以输出目录中的 `status.json` 和 `metrics.jsonl` 为准；只有状态为 `complete` 且存在检查点和渲染文件，才代表训练与导出完成。

当前结果：4000 步及 120 视角导出已完成，产物完整性验证通过。参考视角整图 PSNR 36.00 dB、SSIM 0.959；大角度视角仍有明显模糊和几何失真，尚不适合直接作为高质量扩增样本。详见 `VERIFICATION_ZeroNVS.md`。

## 输入和来源

- 示例：AerialMegaDepth 官方 MASt3R 演示默认使用的 `assets/t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10/00572.jpg`。
- 原图：`inputs/aerialmegadepth/00572.jpg`，2960×3950。
- 训练图：中心裁成正方形并缩放到 256×256；保留整个场景、不抠图、不移除天空，附全不透明 alpha。
- 输入来源、预处理和 SHA256：`inputs/aerialmegadepth/provenance.json`。
- 源码版本及关键文件 SHA256：`source_provenance.json`。
- 相机初始假设：FOV 55°、仰角 20°、尺度 0.7。它们是未经标定的参数，可能影响收敛和场景几何。

## 模型与方法

教师来自 ZeroNVS 官方发布的 `zeronvs.ckpt`。原检查点为 15,466,104,283 字节；下载器通过 HTTP Range 只提取推理所需张量，每个张量先核对官方 ZIP 的 CRC32，再保存为 safetensors。

UNet 的参数取检查点中的 EMA，替换原始 UNet 参数，与官方 guidance 加载逻辑一致。CLIP、VAE、相机条件投影和扩散参数保留官方权重；不使用 Zero123 权重冒充 ZeroNVS。

`scripts/official_guidance.py` 从未修改的官方 `zero123_guidance.py` 直接加载指导类。相机的 19 维条件编码、VAE 编码、各向异性噪声日程和 SDS 损失沿用官方源码；UNet 预测噪声不求导，SDS 梯度经过 VAE 编码器回传到 NeRF 渲染图。采用 PyTorch SDPA 替换显式 attention 矩阵，保持计算公式一致，浮点舍入可能略有不同。

## 与官方默认配置的差异

| 项目 | 本机适配 |
|---|---|
| 哈希网格和体渲染 | 标准 PyTorch CUDA 运算，替代 tiny-cuda-nn / nerfacc 扩展 |
| 几何 | 16 层、每层 2 维哈希特征，独立密度/颜色 MLP，非有界场景收缩 |
| 密度初始值 | 沿用 Magic3D blob 形式 |
| 渲染采样 | 64 个逆深度分层区间；没有官方可训练 proposal 网络 |
| 训练步数 | 4000 |
| 随机视角渲染分辨率 | 前 2000 步为 128×128，后 2000 步为 256×256，与官方分辨率日程一致 |
| 随机视角批量 | 每步 1 个视角，降低官方早期批量为 6 时的显存开销 |
| 参考图监督 | 每步随机 4096 条参考相机射线，是整图 RGB MSE 的抽样估计 |
| 法线平滑 / 朝向正则 | 关闭；不能与原配置的几何约束效果等同 |
| 相机条件 / CFG / SDS | 官方相机编码，CFG=9.5，SDS 权重 0.1，RGB 权重 1500 |
| 显存控制 | 渲染图先计算 SDS 图像梯度，再用完全相同的采样分块重放到 NeRF |
| 导出 | 256×256 的 120 个环绕视角，PNG、MP4 和相机矩阵 |

适配的 NeRF 检查点使用本目录的渲染器加载，与 tiny-cuda-nn 的内部参数布局不兼容。

## 运行

在本目录执行，使用独立的 `.venv-zeronvs` 环境；它读取本机已有 PyTorch 2.9.1+cu130，不修改 AirSim 的 `.venv`。

```powershell
# 后端验证：真实 CUDA 渲染、官方相机编码、分块重放梯度一致性
.\.venv-zeronvs\Scripts\python.exe -X utf8 scripts\check_backend.py

# 仅在尚无权重时执行；支持复用已下载的范围缓存
.\.venv-zeronvs\Scripts\python.exe -X utf8 scripts\inspect_remote_checkpoint.py
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\download_inference_weights.py

# 完整训练与导出
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\run_zeronvs.py

# 从 last.pt 恢复训练
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\run_zeronvs.py --resume

# 用已训练 NeRF 重新导出；不会调用扩散教师
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\run_zeronvs.py --render-only
```

运行参数位于 `configs/aerialmegadepth.json`。中途退出后，恢复会从最近的 250 步检查点继续，随机数状态也保存于检查点。

## 输出检查

默认目录：`outputs/aerialmegadepth_00572/`。

- `last.pt`：NeRF、优化器、训练步数、配置和随机数状态。
- `metrics.jsonl`：每步 SDS、图像梯度、参考图 MSE/PSNR、显存和已用时间。
- `status.json`：加载 / 训练 / 已训练 / 完成的实际状态。
- `reference_step_*.png`：训练中参考视角渲染。
- `renders/*.png`：最终 NeRF 实际渲染的新视角。
- `orbit.mp4`、`contact_sheet.png`：环绕视频与视角总览。
- `transforms.json`：每张渲染图对应的 OpenGL camera-to-world 矩阵、FOV 和分辨率。

参考图 PSNR 只衡量对输入视角的拟合，不是新视角质量指标。当前输入没有配套真值相机轨迹和逐视角真值，不能据此报告新视角 PSNR/SSIM/LPIPS，也不能确认未知表面的真实几何。

## 上游

- https://github.com/kylesargent/ZeroNVS
- https://github.com/kylesargent/zeronvs_diffusion
- https://github.com/kvuong2711/aerial-megadepth

遵循各上游仓库及原始示例数据的许可要求；本目录的来源记录不赋予额外数据授权。
