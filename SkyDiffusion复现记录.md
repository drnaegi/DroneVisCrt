# SkyDiffusion 本地复现记录

日期：2026-10-13。

本轮对齐核验摘要：已确认公开 CVACT 权重曾由官方历史 README 直接链接；当前 Disaster 样本与官方内容一致；新增 Y-SSIM / 截断 PSNR、参数锁定与正式实验预检。详见 `SkyDiffusion对齐核验报告.md`。下文原始运行数字和日志保留，不将评估定义变化当作模型精度提升。

## 实验目标与范围

先验证公开代码和公开 CVACT 权重能否在单张 RTX 4090 上运行，再对已下载的 `SkyDiff/G2A-3` Disaster 子集做小样本跨数据集推理。该实验不等于复现论文 G2A-3 表格；不使用航拍真值作为生成条件。

## 已准备的材料

- 官方仓库：<https://github.com/opendatalab/skydiffusion>，固定提交 `461ccc2fff11882d48f5ab62cae65eef5c45c647`。通过 GitHub API 获取该提交的源码归档，位于 `code/skydiffusion`。
- 论文 PDF：`papers/SkyDiffusion_arxiv_2408.01812.pdf`；可检索文本：同目录下的 `.txt` 文件。
- 新增正式 ICCV 2025 PDF：`papers/SkyDiffusion_ICCV2025_paper.pdf`，表 1 / 表 3 数值已核对。
- 数据：`datasets/G2A-3/{Disaster,History,UAV}.zip`，来源 <https://huggingface.co/datasets/SkyDiff/G2A-3>；下载已完成，三个 ZIP 均通过完整性检查。
- Disaster 的街景与航拍图按相同文件名一一对应，共 2,714 对，分辨率分别为 1024×512 和 512×512。
- 小样本：`datasets/G2A-3/disaster_smoke/pairs.tsv`；从排序后的文件名中用 Python `random.Random(42)` 抽取 8 对。它是流程验证样本，不是作者测试划分。
- 预处理检查图：`datasets/G2A-3/disaster_smoke/preprocessing_preview.jpg`。
- 独立虚拟环境：`envs/skydiffusion`。以 `--system-site-packages` 只读复用系统已有 PyTorch 2.0.0+cu118 / torchvision 0.15.1+cu118，新增依赖安装在虚拟环境内；没有替换 base 中的依赖。
- 推理依赖：`tools/skydiffusion-requirements.txt`。完整环境版本将在实验结束后保存。

## 权重来源与核验标准

公开模型仓库：<https://huggingface.co/SkyDiff1109/SkyDiffusion_ckpt>。

- 文件：`CVACT_Skydiffusion.ckpt`，12,811,886,329 字节。
- 仓库版本：`31b574c1750cc6bb8e285d566b381a0946db9ee2`。
- 发布的 LFS SHA256：`a9108a3394fe127ae7cde9a75019b34ea11da9d3375a51efb4236200bbedfc6a`。
- 本地位置：`checkpoints/SkyDiffusion/CVACT_Skydiffusion.ckpt`。只有下载完成且 SHA256 匹配后，才能视为可用。
- 此仓库的模型卡仅标注 MIT，没有训练配置或实验说明；不据此推断它是作者正式发布的 G2A-3 权重。
- 官方历史 README（提交 `e3ef0b6da5f46875ba24bedf355937daf23518ba`）明确链接该模型仓库，因此 CVACT 权重有官方历史出处；是否对应论文表格最优模型、是否存在 G2A-3 任务权重仍未确认。
- 推理脚本检查所有键名和张量形状，并以 `strict=True` 加载。文本编码器先按本地 CLIP 配置构建，再由完整 checkpoint 填充参数；不以随机参数替代缺失权重。

## 对公开代码的必要修复

修改补丁保存在 `tools/skydiffusion_upstream_fixes.patch`，对应固定源码提交。

1. `skydiffusion_test.py` 缺少逗号，原文件不能通过语法检查；补齐逗号。
2. 推理入口默认 8 GPU / DDP，调整为默认单 GPU，并保留多 GPU 参数；权重加载改为严格检查。
3. `CruvedBEV()` 原来未返回结果；补上 RGB uint8 返回值，使数据集已有的 `/255` 归一化得到 `[0,1]`。
4. 公开投影函数只实现 CVACT/CVUSA 分支；增加 G2A-3 的 1024×512 输入分支。沿用其曲面和北向旋转计算。
5. `FrozenCLIPEmbedder` 增加从本地配置构建的可选方式，避免在完整 checkpoint 已包含文本参数时再次下载整个 CLIP 模型。
6. 新增独立单卡入口 `tools/run_skydiffusion.py`，直接调用官方条件编码器、LAControlLDM 和 DDIM 采样；保存生成图、BEV 图、参数、随机种子、耗时和权重检查结果。
7. 新增评估脚本 `tools/evaluate_skydiffusion.py`，明确使用 RGB PSNR、RGB SSIM 和 AlexNet LPIPS，不使用上游脚本中已失效的 skimage API。
8. 本轮新增可配置曲率、北向偏移、全景尺寸与量化方式；验证默认 UV 和既有 11 张 BEV 图像不变。补丁已重新生成。
9. 评估新增 `--profile upstream`：Y-SSIM 与 40 dB 截断 PSNR；AlexNet LPIPS 在上游 `eval/evaluation.py` 中启用，但其 VAE 重建参考与原始真值的协议差异单独标记。

模型结构已在 CPU 上成功构建：1,460,603,131 个参数，1,576 个 state_dict 键。ControlNet 的输入卷积为 `[320,8,3,3]`，其中 4 通道是噪声潜变量、4 通道是 BEV 编码后的潜变量；RGB 预处理无需人为添加第四通道。

## 实验命令

在 `/root` 下执行：

```bash
/root/DroneVisCrt/envs/skydiffusion/bin/python /root/DroneVisCrt/tools/prepare_disaster_subset.py
/root/DroneVisCrt/envs/skydiffusion/bin/python /root/DroneVisCrt/tools/download_skydiffusion_checkpoint.py
/root/DroneVisCrt/envs/skydiffusion/bin/python /root/DroneVisCrt/tools/run_skydiffusion.py
/root/DroneVisCrt/envs/skydiffusion/bin/python /root/DroneVisCrt/tools/evaluate_skydiffusion.py
```

默认配置：512×512 输出，50 步 DDIM，eta=0，CFG=9，空正/负文本提示，FP32，batch size=1。8 张样本分别使用种子 42–49。

这里沿用公开入口的空文本默认值。公开模型的正/负分支使用相同 BEV 条件；当正负文本也同为空字符串时，两条条件相同，CFG 数值 9 不能理解成额外加强了 BEV 约束。忠实复现还需要核实作者实际训练/评估的条件与提示设置。

网络的系统 DNS 在本次运行中不稳定；`tools/network_dns.py` 为下载脚本提供进程内 DNS-over-HTTPS 解析，不改动系统 DNS。镜像的大文件下载支持断点续传，并在完成后核对 SHA256。CLIP 的配置与词表位于 `checkpoints/clip-vit-large-patch14`，生成过程可离线运行。

## 与论文严格复现之间的差距

- 论文说明 G2A-3 各任务按地域以 4:1 划分训练/测试集。本次下载的 Disaster/History ZIP 未发现划分文件或地域说明。随机抽样不能冒充作者测试集；相邻地点可能存在泄漏。
- 论文 README 指向 `Yejy53/G2A-3`，本次用户截图指定并下载的是 `SkyDiff/G2A-3`。本轮确认 Disaster 5,428 条文件名/大小/CRC 全部一致，当前 16 文件 SHA256 完全一致；不能再把当前 Disaster 质量问题笼统归因于版本差异。History 的 2,128 张街景内容不一致；UAV 未完成内容核验。
- 当前只有标注 CVACT 的公开 checkpoint，尚未核实 G2A-3 各任务专用权重及完整训练配置。
- G2A-3 图像尺寸与北向对齐规则来自论文；但本次沿用公开代码的 0.06 m/pixel 和有符号相机高度 -1.5 m，没有完成 Disaster 的尺度标定。这些参数在运行记录中明确保存，不声称是作者的 G2A-3 参数。
- 当前公开仓库没有完整的顶层训练入口；多街景融合与对应训练/评估准备流程也未形成可直接执行的完整方案。
- 论文采用 8 张 A100、batch size 128、100 epochs 训练。单张 4090 上跑通推理，不代表已复现这个训练过程。
- 8 张图的指标只能检查流程与初步现象；不计算 FID，不用于宣称达到论文精度。指标实现也未确认与作者表格的实现完全一致。

论文表 3 的 Disaster 结果为 FID 96.16、SSIM 0.195、PSNR 13.07、LPIPS 0.685。这些数值仅作为论文记录，不与本次跨数据集小样本实验直接比较。

## 实测结果

论文表 1 / 表 3 与本地两组实测数值的并列对比、差值及实验条件差异，见 [SkyDiffusion论文与当前实验对比](SkyDiffusion论文与当前实验对比.md)。该对比不是同条件正式排名。

### Disaster：8 对跨数据集小样本

已完成下载、权重校验、生成和评估。

| 项目 | 实测结果 |
|---|---|
| checkpoint 校验 | 字节数与 SHA256 均匹配发布的 LFS 元数据 |
| 权重与结构兼容性 | 1,576 / 1,576 个键严格匹配；无缺失键、额外键或形状差异 |
| 图像生成 | 8 / 8 成功，512×512 RGB，无 NaN/Inf |
| PSNR，RGB | 12.135974 dB |
| SSIM，RGB | 0.106019 |
| LPIPS，AlexNet | 0.712021 |
| 平均生成耗时 | 6.83 秒/张，含 BEV 预处理、条件编码、50 步采样与保存 |
| 单次运行总耗时 | 90.46 秒，含 checkpoint SHA256、模型构建及加载 |
| 峰值 GPU allocated 显存 | 7.62 GiB，FP32，batch size=1 |
| FID | 未计算，8 张样本不足以用于有意义的 FID 对比 |

输出目录：`results/skydiffusion_disaster_smoke`。

- `comparison.jpg`：从左至右为街景、Curved-BEV、生成结果、航拍真值。
- `metrics.json`：逐样本指标、均值、指标定义与实验适用范围。
- `metrics_upstream.json`：本轮按公开函数对齐的 Y-SSIM / 截断 PSNR，保留原始真值参考；Disaster 均值为 SSIM=0.104476、PSNR=12.135974、AlexNet LPIPS=0.712021。
- `samples.json`：输入与输出对应关系、种子和单张耗时。
- `run.json`：模型来源、参数、硬件、代码提交与投影假设。
- `checkpoint_compatibility.json`：严格匹配检查。
- `inference.log` / `evaluation.log`：运行日志。

人工检查结果：生成图具有航拍俯视纹理，但多张图的道路、建筑和植被分布与真值明显不一致，并出现棕色空地/郊区偏向。不能把这些图视作对真实灾后区域的可靠重建或损毁预测。运行成功不代表已达到论文的内容一致性；CVACT 权重的域差异、尚未标定的 G2A-3 投影参数以及权重训练信息缺失都是需要进一步核实的因素。

评估基本校验：同一 RGB 图的 SSIM=1、LPIPS=0；PSNR 未使用上游脚本的 40 dB 截断。评估脚本在输入真值尺寸不同时按 OpenCV INTER_LINEAR 缩放至输出尺寸；本次 Disaster 的 8 张真值均原生 512×512，没有触发缩放。

### 仓库自带 CVACT 示例：3 张补充检查

另外使用相同公开 checkpoint 和采样设置，跑通仓库提供的 3 张 CVACT 示例。它们只是示例，不是完整测试集。

- 输出目录：`results/skydiffusion_cvact_examples`；对比图与指标分别为 `comparison.jpg` 和 `metrics.json`。
- 3 / 3 成功生成。RGB PSNR 均值 12.368755 dB，RGB SSIM 0.095622，AlexNet LPIPS 0.641102。
- 本轮对齐公开函数重评估：Y-SSIM=0.097293，截断版 PSNR=12.368755 dB，LPIPS 不变；原始 RGB 报告不覆盖。
- 示例真值原始为 1200×1200，按官方数据加载方式使用 OpenCV 双线性插值缩放到 512×512 后计算指标。
- 人工检查仍观察到明显的道路/建筑分布偏差，因此当前质量问题不能仅归结为 G2A-3 的跨数据集差异。还需要核实该公开权重的训练预处理、准确性与官方代码版本的一致性。

### checkpoint 补充诊断

结果保存在 `results/skydiffusion_checkpoint_diagnostics.json`。

- checkpoint 自带元数据为 epoch=99、global_step=9207、PyTorch Lightning=1.5.9。这是文件内自述，不是训练来源或论文成绩的独立证明。
- ControlNet 的 26 个输出权重/偏置张量均包含非零值，并非整个控制分支仍处于全零输出初始化状态。
- 对种子 42、43、44，分别比较不同街景输入的 Disaster/CVACT 输出，RGB 像素 MAE（0–255 尺度）为 21.56、18.60、13.04，输出不是完全相同。因此不能说模型完全忽略条件；但这些检查也不足以证明它正确地利用了空间布局。

### 当前可复现程度

已经达到“公开模型结构 + 可用公开权重 + 单卡推理 + 小样本跨域定性/定量评估”的程度。尚未达到“论文 G2A-3 正式测试指标复现”，也未开展从头训练或微调。

下一阶段仍需获得作者的 G2A-3 划分与任务权重、Disaster 投影标定、真值参考与 FID 协议。已建立 `configs/skydiffusion_disaster_diagnostic.json` 和 `tools/skydiffusion_preflight.py`：诊断模式可运行，正式 benchmark 模式会拒绝未解决的缺口。若另建本地地域隔离划分，只能标记为自定义实验，不能当作作者正式划分。
