# ZeroNVS NeRF 几何细化开发实验（2026-11-09）

在 RTX 4090 / PyTorch 2.5.1+cu124 上恢复原单双图 2000 步检查点，追加等步数、等总射线预算的几何细化。没有重新调用扩散教师，也没有把第三张照片的 RGB 或预测深度用于细化损失。原始实验目录保持不变。

## 方法及边界

原深度损失约束射线终止距离的均值；前后分散的密度也可能得到正确均值。本实验新增预测表面附近的终止质量约束，以及表面前方的自由空间约束。高置信预测深度仍是 MASt3R 估计值，不是真值。

每步总计 2048 条训练射线、每条 96 个区间。单图仅使用 00572，双图使用 00572 和 00563；每组从各自原始检查点开始，重建 Adam 优化器，学习率 0.001。RGB 权重 1500，平均深度权重 20。表面带宽为预测距离的 8% 加 0.02。每个配置只评估固定最终步，不按评估图早停。

共享相机估计使用了全部三张图。00616 已经在历史实验和本次配置探索中被反复检查，因此这里属于开发集诊断，不是独立测试。只有一个场景和一个随机种子；不能宣称普遍的双图优势。原模型复测与归档结果的最大 PSNR 差约 0.00023 dB。

## 全部探索结果

下表全部为 00616 的 PSNR / SSIM。各配置对应同名 JSON，结果目录在 outputs/ 下，带 20261109 后缀。

| 配置 | 追加步数 | 表面 / 自由空间权重 | 单图 | 双图 |
|---|---:|---:|---:|---:|
| 原始归档复测 | 0 | — | 13.6475 / 0.20385 | 12.2421 / 0.26738 |
| surface_refinement | 600 | 20 / 20 | 12.7253 / 0.17559 | 13.5215 / 0.28020 |
| surface_refinement_weak | 600 | 2 / 2 | 13.4630 / 0.19513 | 12.5832 / 0.26898 |
| surface_refinement_free_space | 600 | 2 / 20 | 13.1759 / 0.19737 | 12.3313 / 0.26559 |
| surface_refinement_long | 2000 | 20 / 20 | 12.8197 / 0.17364 | 13.8618 / 0.27784 |

2000 步配置双图相对原双图提升 1.6196 dB / 0.01045 SSIM；相对原单图提升 0.2142 dB / 0.07398 SSIM；相对同样追加 2000 步的单图提升 1.0421 dB / 0.10419 SSIM。

这里的改进很有限。单图细化本身退化；双图训练视角拟合也下降（地面 PSNR 29.8288 → 28.7553，俯视 32.5591 → 31.4314）。第三视角仍被大片模糊密度遮住，尚未恢复可用建筑结构。因此只能报告本开发视角两项指标超越原单图，不能称为高质量新视角生成成功。没有做等步数但不新增约束的续训消融，不能把全部增益归因于新增损失；也未做多种子检验。

## 复现

使用可用 CUDA 的 PyTorch 环境，并安装 numpy、Pillow、imageio、imageio-ffmpeg、roma。此次环境为 `.venv-refine`，使用系统 PyTorch，额外依赖安装在该虚拟环境中。

先在仓库根目录运行 `python sample_augmentation/scripts/restore_multiview_checkpoints.py`。下载器通过 GitHub CLI 读取 Release，核验 ZIP 和解压文件的 SHA256，恢复到独立的 `checkpoint_downloads/restored/`，不覆盖正式实验输出。

然后在仓库根目录执行（输出目录必须尚不存在）：

```bash
python sample_augmentation/scripts/refine_multiview_geometry.py \
  --source sample_augmentation/checkpoint_downloads/restored/outputs/multiview_a10 \
  --config sample_augmentation/configs/surface_refinement_long.json \
  --output sample_augmentation/outputs/surface_refinement_long_retest
```

输出包含 protocol.json、单双图逐步 metrics.jsonl、status.json、last.pt、全部三视角渲染、comparison.png 和 results.json。协议记录源文件摘要、训练图片 ID、配置和运行设备。检查点遵循现有忽略规则，不自动入库。

## 验证与下一步

三个针对性单元测试验证：正确均值下仍能识别分散的双层密度；无效深度不产生监督；梯度抑制近处遮挡并增加表面质量。另已通过 CUDA 实际渲染和反向传播检查。

后续应先增加同预算无新增约束的续训对照，核验相机与预测深度的跨视角可见性，再使用更多未参与开发的新视角和随机种子验证。不能通过进一步反复调同一张 00616 的指标来证明泛化。
