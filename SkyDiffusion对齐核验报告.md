# SkyDiffusion：权重、投影、划分与评估对齐核验

更新日期：2026-10-13。核验对象为官方代码提交 `461ccc2fff11882d48f5ab62cae65eef5c45c647`、已下载的 CVACT checkpoint，以及当前 Disaster 8 对 / CVACT 3 对诊断样本。

## 1. 总结

| 项目 | 已完成 | 仍不能确认 |
|---|---|---|
| 权重 | 找到官方仓库历史 README 的下载链接；本地 SHA256、字节数与 1,576 个结构键匹配 | 当前是 CVACT 权重，不是已确认的 Disaster/UAV/History 表格权重；训练配置未随文件发布 |
| 投影 | 明确尺度、高度、曲率、北向偏移与全景尺寸；默认 UV 与原始公开实现完全一致，11 张既有 BEV 图像像素不变 | Disaster 训练时的尺度/高度/曲率和论文公式的坐标约定 |
| 正式划分 | 完成 2,714 对 Disaster 清单，保留文件名坐标、ZIP 成员名、大小、CRC；核验官方与已下载版本 | 作者按地域 4:1 的具体 train/test IDs 与地域边界；不能用随机 8 张替代 |
| 评估 | PSNR 与 Y-SSIM 逐样本数值等于两个公开脚本的函数；确认一个公开脚本启用 AlexNet LPIPS | 论文使用原始真值还是 VAE 重建真值、PNG 量化方式和 FID 的实际实现/统计集 |

**当前仍是流程复现和诊断实验，不是论文正式指标复现。** 未在这 8 张数据上调参追分，也未创建一个“作者正式划分”的替代品。

## 2. 权重：官方历史出处已确认，但任务并未对齐

### 官方下载链路

官方仓库在提交 `e3ef0b6da5f46875ba24bedf355937daf23518ba` 的 README 中明确指向当前下载仓库：

- [历史官方 README](https://github.com/opendatalab/skydiffusion/blob/e3ef0b6da5f46875ba24bedf355937daf23518ba/README.md)。其中说明可从 `SkyDiff1109/SkyDiffusion_ckpt` 下载 CVACT 预训练模型。
- [checkpoint 仓库](https://huggingface.co/SkyDiff1109/SkyDiffusion_ckpt/tree/31b574c1750cc6bb8e285d566b381a0946db9ee2)。本地文件大小为 12,811,886,329 字节。
- SHA256：`a9108a3394fe127ae7cde9a75019b34ea11da9d3375a51efb4236200bbedfc6a`。
- 历史 README 与提交 API 快照保存在 `results/alignment_audit/sources/historic_README.md` 和 `historic_code_commit.json`。

因此应修正早先过于宽泛的来源疑虑：**不是完全无官方出处的第三方权重；它曾由官方仓库直接链接。** 但这仍不能证明它对应 ICCV 表 1 的 CVACT 最优结果，更不能当作 G2A-3 各任务专用权重。

### 文件内证据

`results/alignment_audit/checkpoint_metadata.json` 保存了以下记录：

- epoch=99、global_step=9207、PyTorch Lightning=1.5.9。
- 回调中的内部保存路径带有 `slurm_scripts/cvact/results/202405030330/`。这是文件内自述，支持 CVACT 训练标签，不是独立成绩证明。
- 未包含 `hyper_parameters` 或完整训练/投影配置。优化器记录有 learning rate=1e-5、weight decay=0.01。
- 条件编码器的 108 个张量与第一阶段 VAE 编码器/quant_conv 相等；对应代码中冻结、复制 VAE 编码器的设计，不应据此断定 checkpoint 未训练。

检索了官方仓库 releases、issues、公开模型搜索与作者账号模型列表，未找到可核验的 G2A-3 三任务专用 diffusion checkpoint。作者账号中的 BEV 检索模型不能冒充 diffusion 权重。Issue #4 中存在多位用户关于该 CVACT 模型生成质量的相似反馈；这属于复现风险线索，不是已确定的故障原因。

## 3. 投影：保留公开默认值，显式暴露未知量

| 参数 | 当前 Disaster 锁定值 | 依据 / 状态 |
|---|---:|---|
| 输出边长 | 512 | 论文与公开实现一致 |
| 街景输入 H×W | 512×1024 | ICCV 正文中的 G2A-3 尺寸；本地街景原生相同 |
| `meter_per_pixel` | 0.06 | 公开代码默认；未证明适用于 Disaster 航拍覆盖范围 |
| `camera_height` | −1.5 | 公开代码使用有符号高度，不要改成 +1.5 后仍称相同配置 |
| `curve_height` | 3.0 | 公开代码曲面乘数；不能声称正文公布了单视图 λ=3 |
| `north_offset_deg` | 0 | 保留公开默认方向，不对测试样本调偏移 |
| 曲面 | `curve_height × normalized_radius^4` | 与原始公开代码一致 |

实现现在支持 `--curve-height`、`--north-offset-deg`、`--panorama-height`、`--panorama-width` 等参数，保存于运行记录。它们是可配置入口，不是已获作者确认的标定值。

### 两个不能盲改的约定

1. 公开代码写 `rot=90`，后面却乘以 π；即 90π，等于 45 个完整周期，而不是常见的 90°。现有代码已经让北、东、南、西分别映射到约 W/2、3W/4、0、W/4。直接“修成 90°”会平白引入四分之一全景的方向变化。新增偏移以真正的角度换算叠加，但默认严格保持不变。
2. 正文式 (2) 与公开实现的垂直方向存在坐标/符号约定差异。不能仅凭排版公式翻转全景垂直坐标；需要确认训练时究竟使用哪一版实现。

另有尺寸差异：正文统一说明 512×1024，但公开 CVACT/CVUSA 分支分别重采样到 832×1664 / 616×1232。两者宽高比相同，仍可能因插值产生条件输入差异。此前 3 张 CVACT 示例保持公开分支默认，不声称该尺寸已与 checkpoint 训练预处理对齐。

验证结果位于 `results/alignment_audit/projection_validation.json`：默认 UV 与固定上游版本按位相等；已有 11 张 BEV 图像像素完全一致；四个方位的误差小于 1 个全景像素，额外 +90° 偏移的误差小于 0.001 像素。

## 4. 数据版本与正式划分

### Disaster：当前使用的数据内容已得到更强验证

官方 README 指向 [Yejy53/G2A-3](https://huggingface.co/datasets/Yejy53/G2A-3)，下载的数据来自 [SkyDiff/G2A-3](https://huggingface.co/datasets/SkyDiff/G2A-3)。官方核验版本为 `ec02e06844677f990fb62dc1e63a9a46051b0eaf`。

- 两个 Disaster ZIP 的容器 SHA256 不同，但 **5,428 个图像条目的文件名、大小和 CRC 全部一致**。
- 对当前 8 对街景/航拍共 16 个文件额外读取官方内容，逐个 SHA256 与本地完全一致。
- 不必因为 ZIP 容器哈希不同就重下载整套 Disaster，也不能再把这 8 对的质量偏差笼统归因于发布版本不同。
- CRC 并非密码学哈希；尚未逐个对全部 5,428 文件进行 SHA256 内容比较。证据范围应如实区分。

报告：`results/alignment_audit/disaster.zip.audit.json`。

### History：确实存在街景内容差异

- 2,128 个航拍图像的文件名、大小、CRC 一致。
- 街景目录名称不同，但不只是改目录：按 basename 对齐后，2,128 张街景的大小/CRC 仍不一致。
- 固定种子抽检的 3 张同名街景，分辨率同为 2048×1024，但实际解码像素不同，RGB MAE（0–255）约 45.89、34.42、78.37。
- 未证明差异仅是 JPEG 重新编码或北向旋转；后续做 History 正式实验应优先使用官方版本并重新核验。

报告：`results/alignment_audit/History.zip.audit.json`。只按需读取远端 ZIP 元数据和选定图像，没有再次下载整个 History/UAV 数据集。UAV 的外层 ZIP 大小不同，官方嵌套图像内容尚未完成核验。

### 正式 train/test 仍是缺口

ICCV 正文明确要求各 G2A-3 任务按地域约 4:1 划分。已核验的官方/历史代码树、Disaster/History ZIP 条目与主页中未找到可直接使用的正式 IDs 或地域边界；这表示“在已检索材料中未找到”，不是证明作者从未提供过。

新增 `datasets/G2A-3/disaster_inventory.tsv`，包含全部 2,714 对的 ID、文件名中解析的经纬度、ZIP 条目、大小、CRC，`official_split` 全部标记为 `unknown`。文件名坐标未独立地理定位。该清单不是推理入口的三列 TSV，也不是正式划分。

没有随意按经纬度挑一个边界、没有随机切 80/20，也没有在当前 8 张上做验证集调参。划分核验摘要：`results/alignment_audit/split_status.json`。

## 5. 评估函数已对齐，论文评估协议尚未完全对齐

公开仓库中存在多套评估入口，必须分开看：

| 项目 | `eval/eval.py` | `eval/evaluation.py` | 当前对齐后的本地评估 |
|---|---|---|---|
| PSNR | RGB float32 MSE，eps=0.0001，上限 40 dB | 相同 | `psnr_rgb_capped40_db`，逐样本等于公开函数 |
| SSIM | RGB→YUV 后的 Y 通道；默认 7×7 均匀窗口 | 相同 | `ssim_y`，逐样本等于公开函数 |
| LPIPS | 初始化和调用被注释 | **启用 AlexNet**，输入 RGB [-1,1] | AlexNet LPIPS 0.1.4；网络类型、归一化相符 |
| 比较真值 | 由调用者给定目录 | `*_reconstruction.png`：VAE 重建真值 | 历史结果用原始真值；新增可选重建参考分支并单独存报告 |
| 输出量化 | 已保存的文件 | 上游 logger 转 uint8 时截断 | 历史结果四舍五入；新增 `--quantization upstream` 支持上游方式 |

因此，不能泛称“上游没有 LPIPS 实现”。也不能只改成 Y-SSIM 就宣布论文评估口径已全部对齐。

`tools/evaluate_skydiffusion.py --profile upstream` 对已有输出重算指标，保留原来的 `metrics.json`，另写 `metrics_upstream.json`：

| 诊断集 | 论文 SSIM | 旧 RGB SSIM | 对齐公开函数的 Y-SSIM | PSNR（40 dB 截断版） | AlexNet LPIPS |
|---|---:|---:|---:|---:|---:|
| Disaster 8 对 | 0.195 | 0.106019 | **0.104476** | 12.135974 | 0.712021 |
| CVACT 3 张示例 | 0.118 | 0.095622 | **0.097293** | 12.368755 | 0.641102 |

这次是**评估定义修正，不是模型精度提升**。既有生成图和真值不变；新 SSIM 仍与论文有差距，且样本、模型任务和正式划分不一致。

公开 `eval/fid.py` 含硬编码路径、单图协方差等问题，不能把它当作可用的标准 FID 实现。另一脚本末尾留有 `python -m pytorch_fid ...` 的注释，但未锁版本、Inception 权重或正式评估名单。暂不计算 8/3 张的小样本 FID，也不把坏脚本返回值当成绩。

正式 ICCV PDF 已保存为 `papers/SkyDiffusion_ICCV2025_paper.pdf`；表 1 / 表 3 的上述论文数值与此前 arXiv 对比一致。[CVF 论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Ye_Leveraging_BEV_Paradigm_for_Ground-to-Aerial_Image_Synthesis_ICCV_2025_paper.html) 未找到链接的补充材料，尝试的常规 supplemental URL 返回 404；不据此声称所有渠道都没有补充材料。

### 锁定配置重跑与 VAE 参考诊断

新增独立结果目录 `results/skydiffusion_alignment_locked`。同样的 8 个输入和种子下，生成图与原始运行的 8 张图像像素完全一致；另保存 8 张 VAE 重建参考，未把航拍真值作为生成条件。逐图验证记录为 `results/alignment_audit/locked_run_validation.json`。

| 参考真值 | PSNR ↑ | Y-SSIM ↑ | AlexNet LPIPS ↓ |
|---|---:|---:|---:|
| 原始 target（同前） | 12.135974 | 0.104476 | 0.712021 |
| VAE 重建 target（独立诊断） | 12.163325 | 0.112185 | 0.702998 |
| 论文 Disaster（参考协议未确认） | 13.07 | 0.195 | 0.685 |

VAE 参考令指标小幅变化，但仍未弥合差距。这是**评价参考变了，不是生成模型变好了**。当前仍按先采样、后独立 VAE 重建保存，使用已有 PNG 四舍五入方式；不能声称完整复刻了上游 logger 的 RNG 顺序与截断量化协议。两种参考的报告分开保存，没有覆盖旧结果。

## 6. 锁定配置、预检与下一步

锁定的诊断配置：`configs/skydiffusion_disaster_diagnostic.json`。记录权重 revision/hash、输入清单 hash、代码文件 hash、投影、50 步 DDIM、CFG=9、空正/负提示、种子、精度、量化方法与评估参考。它**不是官方 Disaster 配置**。

在项目目录下使用当前环境：

```bash
envs/skydiffusion/bin/python tools/validate_skydiffusion_alignment.py
envs/skydiffusion/bin/python tools/skydiffusion_preflight.py --mode diagnostic
envs/skydiffusion/bin/python tools/skydiffusion_preflight.py --mode benchmark
envs/skydiffusion/bin/python tools/run_skydiffusion.py --profile configs/skydiffusion_disaster_diagnostic.json --preflight-only
```

- `diagnostic` 预检通过：本地文件/hash 与已锁定配置一致，但始终保留四类 benchmark 缺口。
- 锁定配置默认输出到 `results/skydiffusion_alignment_locked`，不覆盖原来的 smoke 结果；运行目录另存 `locked_profile.json`，避免后续配置更新丢失当次配置。
- `benchmark` 必须拒绝当前配置，退出码 2；不允许把 CVACT 跨域诊断与未知划分自动升级为正式实验。
- 脚本的校验只能核对输入证据和锁定配置，不能凭人为修改 `verified` 字段来证明作者来源。
- `validate_skydiffusion_alignment.py` 会按现有代码更新本地锁定 hash，用于有意更新配置，不是用来隐藏变更。正式运行前应保留配置文件与结果一起归档。
- 如需 VAE 参考诊断：运行入口使用 `--save-reconstruction-reference`，评估入口使用 `--reference vae_reconstruction`；航拍真值只参与推理后的独立重建参考生成，不进入 BEV 条件或扩散采样。重建分支另写 `metrics_upstream_reconstruction.json`，不得覆盖原始真值结果或宣称是论文口径。

没有取得缺失材料前，不做昂贵全量训练或基于测试图追分的曲率/方向扫描。下一阶段应向作者索取：

1. ICCV 表 3 各任务的 checkpoint 及配置、准确代码提交和训练时 BEV 预处理版本。
2. Disaster/History/UAV 的正式 train/test IDs、地域边界及数据发布 revision。
3. 各任务的地图尺度、相机高度、单视图 λ、北向和垂直符号约定；是否使用论文统一尺寸或公开分支尺寸。
4. 指标究竟对原始 target 还是 VAE reconstruction 计算；LPIPS/SSIM/PNG 量化版本，FID 命令、依赖版本与参考统计集。

一旦拿到上述材料，优先补齐配置与划分，再运行完整正式测试；在此之前，当前结论仍保持“流程可运行，论文成绩未复现”。
