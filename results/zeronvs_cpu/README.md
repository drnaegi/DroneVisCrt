# ZeroNVS CPU 新视角合成复现

本次使用 MegaScenes 官方发布的 ZeroNVS 微调模型（iteration 90000），不是 AerialMegaDepth 空地微调权重。
原版 ZeroNVS 官方 Google Drive 链接在本次访问中返回下载配额限制。
当前目标为单图扩散生成，不包括 NeRF/SDS 蒸馏或论文的 DTU/Mip-NeRF 360 指标评测。

## 已完成结果

两个新视角均已通过真实权重 CPU 推理生成，输出为 256×256 RGB。

| 项目 | 实测 |
| --- | --- |
| CPU | Intel Xeon Silver 4314，16 个 PyTorch 线程 |
| 模型加载 | 首次进程 12.22 秒；独立重跑 11.25 秒 |
| -10° 环绕视角 | 51.06 秒 |
| +10° 环绕视角 | 51.42 秒 |
| 独立重跑进程峰值 RSS | 10.11 GiB，包含加载与推理 |
| 精度和采样 | FP32、DDIM 配置 50 步（官方实现实际 49 次更新）、eta 1、guidance 3、seed 42 |

计时包含去噪和 VAE 解码，不包括模型加载及参考图片编码。生成张量均为有限值。
`comparison.png` 左侧为输入，中间为 -10°，右侧为 +10°。可观察到建筑透视变化，但窗户、树木、自行车、路面等细节也有改动，不能作为真实视角的精确重建。
没有目标视角真值，因此没有报告 PSNR/SSIM 等精度指标。

第一次进程已保存 -10° 图片，随后在 +10° 生成中被外部终止；-10° 时间从日志恢复，精度为 0.01 秒。
+10° 已单独重新运行成功，完整运行元数据在 `right/run.json`；汇总在 `summary.json`。

## 运行

在项目目录的 Bash 终端执行：

```bash
envs/zeronvs-cpu/bin/python tools/run_zeronvs_cpu.py --steps 50 --threads 16 --angles -10 10
```

默认输入是之前 AerialMegaDepth 测试中的地面建筑照片，中心裁剪到 256×256。
可以通过 `--image /absolute/path/input.jpg --output results/zeronvs_custom` 更换输入和保存位置。
FOV 60°、参考深度 20% 分位数为 1、围绕参考坐标系 `(0,0,-1)` 旋转均为演示假设，不是真实相机标定。
OpenGL camera-to-world 约定、平移归一化和 19 维相机编码已与官方 `compute_T` 核对。

## CPU 适配

- 使用独立环境 `envs/zeronvs-cpu`，继承基础 PyTorch，未修改已有模型环境。
- 保留官方 UNet、VAE、CLIP 预处理、DDIM 去噪和 classifier-free guidance。
- 通过 DDIM 子类将固定 CUDA 缓冲区改为模型实际设备。
- 直接构造同结构 CLIP ViT-L/14，再严格加载权重中自带的全部参数，避免重复下载 CLIP。
- 使用非 EMA 参数，与 MegaScenes 官方 `use_ema_scope: False` 一致。
- 预检通过：1,250 个非 EMA 张量的键名和形状与官方文件头匹配。
- DDIM 50 步、eta 1、guidance scale 3、CPU FP32；不同于原版 ZeroNVS 完整蒸馏评测。
- MegaScenes 官方采样器默认 `t_start=-1` 会去掉采样时间步列表最后一项；保留该行为，配置 50 步时实际执行 49 次去噪更新，并在结果中记录实际次数。

## 文件

- `tools/run_zeronvs_cpu.py`：运行入口。
- `tools/zeronvs_cpu_support.py`：CPU 与本地 CLIP 加载适配。
- `tools/download_zeronvs_ms.py`：可续传权重下载器。
- `tools/extract_zeronvs_inference.py`：只补齐推理所需范围，原样提取 1,250 个非 EMA 张量，保存为 `model-inference.safetensors`；不转换精度或更改数值。
- `source_manifest.json`：源码固定提交和实际文件 blob SHA-1。
- `environment.freeze.txt`：可见环境依赖。
- 成功执行后生成 `input.png`、`orbit_-10.png`、`orbit_+10.png`、`comparison.png`、`run.json`。
- 本次实际保存：`orbit_-10.png`、`right/orbit_+10.png`、根目录 `comparison.png`、`summary.json` 及 `right/run.json`。

## 来源

- 原版：https://github.com/kylesargent/ZeroNVS
- 原版扩散代码：https://github.com/kylesargent/zeronvs_diffusion ，提交 `f82b71a29ad7497b8d4b6bbbcd8aea44ed5c43a9`。
- 本次对应代码：https://github.com/MegaScenes/nvs ，提交 `0de3b004f643e0584112e7282a14ec304358e93f`。
- 权重：https://s3.us-west-2.amazonaws.com/megascenes/nvs_checkpoints/zeronvs_finetune/iter_90000/model.safetensors
- 文件大小：8,583,857,168 字节；S3 multipart ETag：`cfd91b18bbaba0026594e90359a60c37-164`。

源码为经过 blob 哈希核对的部分快照，未下载大数据和全部演示资源。
本次采用推理参数提取方式，跳过约 3.44 GB EMA 副本。HTTP 范围响应验证源 ETag、Content-Range 和字节数，再为本地完整推理文件生成 SHA-256。
由于没有下载完整原文件，不宣称重算通过原文件的 multipart ETag；具体来源和校验范围见权重目录的 `inference_manifest.json`。
完整下载器仍可续传并校验原文件。原始 `.part` 文件保留续传进度，不能用作完成权重。
