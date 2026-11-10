# Lego 与 LLFF Fern 稀疏视角诊断

目的：将原建筑实验的相机估计、预测深度和扩散教师因素暂时移开，先在 Lego 准确相机上检查现有 NeRF 后端是否能从第二张参考图中获益，再用 LLFF Fern 真实照片验证同一固定配置。

这是已有哈希网格 NeRF 的纯 RGB 基线排查，不是 ZeroNVS SDS 的完整复现，也不直接检验前一轮的表面约束损失。先保留可解释的基础对照，后续教师/深度改动需要单独对照。

## 本次结果

两场景共 28 组已完成，完整指标、逐种子差值、固定测试视角对照及实际参考输入见 [实验报告](outputs/nerf_diagnostic_20261109/RESULTS.md)。主实验三个种子的测试均值：Lego 单图 → 双图 PSNR 为 12.65 → 14.45 dB，SSIM 为 0.539 → 0.565；Fern 为 18.82 → 20.94 dB、0.555 → 0.684。每个种子的两项平均指标均提高。

这不等于逐视角、逐指标全面改善：Lego 每个种子的 20 个测试视角 PSNR 均提高，但 SSIM 分别只在 14/13/13 个视角提高；Fern 三个种子的 3 个测试视角均在两项指标上提高。Lego 双图结果仍有明显拉伸和错误几何，八图能恢复更完整的形状但仍有噪声；Fern 双图改善较大视角偏移处的结构，八图细节更清楚。本次结果说明后端能够利用新增视角，不能证明原建筑实验已修复，也不能证明两图足以完成高质量全视角重建。

## 数据

官方源：<https://github.com/bmild/nerf/blob/master/download_example_data.sh>。

下载 `https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip`，实际 370385516 字节，SHA256 `ce4e94e031c099a19ef04cfb6c71f1e47225d97d365be610b476e379a386c25f`。这个摘要是本次获取后计算的记录，不是发布方签名或预公布校验值。ZIP CRC 和逐文件 SHA256 均记录；原始数据遵循上游许可。

- Lego：100 张训练、100 张验证、200 张测试渲染图，精确相机。训练池只使用官方 train；开发固定每 10 张 val 取 1 张；最终测试固定每 10 张 test 取 1 张。实验分辨率 128×128，先合成白底再缩小，不用 alpha 训练监督。
- Fern：20 张真实照片。使用官方排序和 SfM 相机；索引模 8 为 0 的 3 张用于最终测试，为 4 的 2 张用于开发，其余 15 张为训练池。128×96，不裁剪；相机轴向按原 NeRF LLFF 读取方式转换。
- 分组先于训练冻结；所选输入与开发、测试没有相同文件摘要。LLFF 的 SfM 相机由上游联合估计，不能宣称仅凭一两张照片独立获得相机。
- 候选训练相机首张固定为排序首张，第二张优先选择相对旋转约 20° 的视角，其余按相机特征最远点扩展；不看图像质量或测试分数选参考。Lego 第二张约 19.69°，Fern 第二张约 12.38°。

## 实验矩阵

每场景 14 组：分层采样下 1/2/4/8 图 × seeds 0/1/2；均匀采样下 1/2 图 × seed 0。每组 1500 步、每步总计 1024 条射线和每条 64 次场查询，学习率 0.005 → 0.0005，hash_log2_size=17。多图均分总射线预算。新增参考不会获得额外训练步数。不同采样的 GPU 时间不保证完全相同，实际运行时间单独记录。

均匀采样为 64 个分层随机位置。重要性采样先查询 32 个位置，依据其终止权重再查询 32 个位置，合并排序后积分并复用粗采样的场输出；不额外调用教师。渲染使用固定分位数，训练使用固定种子的随机分位数。

Lego 的 z 参数范围 2–6；Fern 的近远边界及场景归一化由训练池相机元数据计算。射线方向被归一化后，z 参数范围按方向长度转换为欧氏距离。这里不采用 LLFF 原版 NDC，且现有颜色网络不含观察方向输入；这些均为本地后端的能力边界，不可称为原论文复现。

只保存并评价固定最终步。每场景只有全部预定组完成后才允许打开最终测试，保存 test_seal.json 与检查点哈希；测试后禁止继续往同一实验目录追加训练。

## 运行

环境使用现有 `.venv-refine`（可用 CUDA 的 PyTorch、numpy、Pillow）；汇总绘图使用 matplotlib。以下命令在仓库根目录运行。

```bash
sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/download_nerf_diagnostic_data.py

sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/run_nerf_diagnostic.py \
  --scene lego --output sample_augmentation/outputs/nerf_diagnostic_20261109/lego --stage prepare
sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/run_nerf_diagnostic.py \
  --scene fern --output sample_augmentation/outputs/nerf_diagnostic_20261109/fern --stage prepare
```

先对 Lego 执行以下训练命令；读取开发结果并确认数据/相机/数值检查后，再以相同参数对 Fern 执行（替换 scene 与输出目录）：

```bash
sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/run_nerf_diagnostic.py \
  --scene lego --output sample_augmentation/outputs/nerf_diagnostic_20261109/lego
sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/run_nerf_diagnostic.py \
  --scene lego --output sample_augmentation/outputs/nerf_diagnostic_20261109/lego \
  --sampler uniform --counts 1 2 --seeds 0
```

两场景训练完成后分别执行最终测试，再汇总：

```bash
sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/run_nerf_diagnostic.py \
  --scene lego --output sample_augmentation/outputs/nerf_diagnostic_20261109/lego --stage test
sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/run_nerf_diagnostic.py \
  --scene fern --output sample_augmentation/outputs/nerf_diagnostic_20261109/fern --stage test
sample_augmentation/.venv-refine/bin/python sample_augmentation/scripts/summarize_nerf_diagnostic.py \
  --root sample_augmentation/outputs/nerf_diagnostic_20261109
```

已完整完成的训练组在协议与源代码摘要完全一致时跳过；部分完成组或配置变化会报错，需要使用新输出目录。没有自动覆盖原研究结果。

## 产物与指标

每组保存协议、训练日志、模型检查点、训练/开发/测试渲染和逐视角指标。PSNR 先按视角计算，再等权平均；SSIM 采用 7×7 均匀窗口、样本协方差和完整有效窗口。Lego 额外用测试图 alpha≥0.5 报告前景 PSNR，避免白底主导；alpha 不用于训练。主表为三个种子的均值和样本标准差，并列出双图减单图的逐种子差值；只有一个种子的采样消融单独标注。

结果入口为 `outputs/nerf_diagnostic_20261109/RESULTS.md`、`summary.json`、`view_count_psnr.png`，两场景各有 `test_comparison.png`。图示使用固定最前几个测试视角，未挑选最好结果。原数据放在被 Git 忽略的 `datasets/`，检查点也按现有规则忽略。

## 检查

```bash
sample_augmentation/.venv-refine/bin/python -m unittest discover -s tests -p test_nerf_diagnostic.py -v
```

测试覆盖 LLFF 坐标转换、归一化射线的深度保持、空密度 PDF、常密度积分解析解与嵌套相机选择。均匀和分层采样均已通过实际 CUDA 渲染/反传检查。若模型在训练视角表现很好但留出视角失败，应报告几何欠约束，不能把训练拟合替代新视角能力。
