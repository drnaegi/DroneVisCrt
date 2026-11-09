# NeRF 检查点下载与恢复

全部 5 个 NeRF 检查点已归档到私有仓库的 [nerf-distillation-20261109 Release](https://github.com/drnaegi/DroneVisCrt/releases/tag/nerf-distillation-20261109)。每个原始 `last.pt` 约 192 MiB，超过普通 Git 单文件限制，因此通过 Release 附件保存。访问权限与私有仓库一致。

## 归档内容

| Release 附件 | 解压后的检查点或相机文件 | 实际保存步数 | 用途 |
| --- | --- | ---: | --- |
| `zeronvs-single-4000.zip` | `outputs/aerialmegadepth_00572/last.pt` | 4000 | 正式单图蒸馏 |
| `multiview-single-2000.zip` | `outputs/multiview_a10/single/last.pt` | 2000 | 正式对照实验：单图组 |
| `multiview-dual-2000.zip` | `outputs/multiview_a10/multi/last.pt` | 2000 | 正式对照实验：双图组 |
| `smoke-single-2.zip` | `outputs/smoke_00572/last.pt` | 2 | 冒烟测试 |
| `aborted-inverse-single-1000.zip` | `outputs/multiview_a10/aborted_inverse_sampling/single/last.pt` | 1000 | 中止的旧采样实验 |
| `multiview-calibration.zip` | `outputs/multiview_a10/cameras.pt`、`cameras.json` | — | 多图相机、内参和预测深度 |

每个 NeRF 归档还包含对应目录已有的 `config.json`、`status.json`。中止实验的状态日志记录到 1250 步，但实际最后保存的检查点是 1000 步；以检查点为准。冒烟测试和中止实验均不参与正式对照结论。

检查点保留 NeRF 参数、优化器状态、配置和步数；单图蒸馏检查点另含随机数状态，多图检查点另含相机配置标识。它们使用本模块的 PyTorch 渲染器，与 tiny-cuda-nn 的内部参数布局不兼容。

## 下载、核验与解压

可在 Release 页面登录 GitHub 后下载全部 6 个 ZIP，以及 `CHECKPOINTS_MANIFEST.json` 和 `SHA256SUMS.txt`。如果安装并登录了 GitHub CLI，也可在本模块目录执行：

```powershell
gh release download nerf-distillation-20261109 --repo drnaegi/DroneVisCrt --dir checkpoint_downloads
```

以下命令在 `sample_augmentation/` 下执行；它先用 Git 中的 [CHECKPOINTS_MANIFEST.json](CHECKPOINTS_MANIFEST.json) 检查全部 ZIP 的大小和 SHA256，再解压到本模块。解压会覆盖归档中的同名检查点、配置和状态文件；如已开始新实验，请先备份对应输出目录。

```powershell
$checkpointManifest = Get-Content .\CHECKPOINTS_MANIFEST.json -Raw | ConvertFrom-Json
foreach ($checkpointAsset in $checkpointManifest.assets) {
    $checkpointArchive = Join-Path .\checkpoint_downloads $checkpointAsset.name
    if ((Get-Item -LiteralPath $checkpointArchive).Length -ne $checkpointAsset.size_bytes) {
        throw "Size mismatch: $($checkpointAsset.name)"
    }
    $checkpointHash = (Get-FileHash -LiteralPath $checkpointArchive -Algorithm SHA256).Hash
    if ($checkpointHash.ToLowerInvariant() -ne $checkpointAsset.sha256) {
        throw "SHA256 mismatch: $($checkpointAsset.name)"
    }
}
foreach ($checkpointAsset in $checkpointManifest.assets) {
    Expand-Archive -LiteralPath (Join-Path .\checkpoint_downloads $checkpointAsset.name) -DestinationPath . -Force
}
foreach ($checkpointAsset in $checkpointManifest.assets) {
    foreach ($checkpointFile in $checkpointAsset.files) {
        $checkpointHash = (Get-FileHash -LiteralPath $checkpointFile.path -Algorithm SHA256).Hash
        if ($checkpointHash.ToLowerInvariant() -ne $checkpointFile.sha256) {
            throw "Extracted SHA256 mismatch: $($checkpointFile.path)"
        }
    }
}
```

## 重新渲染

在已配置 CUDA 环境中，正式单图检查点可直接重新导出；多图评估需同时恢复两个正式检查点与 `multiview-calibration.zip`。下面两条命令都不加载扩散教师：

```powershell
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\run_zeronvs.py --render-only
.\.venv-zeronvs\Scripts\python.exe -X utf8 -u scripts\run_multiview_experiment.py --mode evaluate
```

恢复训练仍需要相应教师权重。中止实验使用旧逆深度采样，不能直接视为当前正式多图配置的可续训检查点。

本次附件不包含教师权重或 `pair_predictions.pt`。历史完整性报告反映原工作区的运行结果；新视角模糊、失真等质量限制仍然成立。
