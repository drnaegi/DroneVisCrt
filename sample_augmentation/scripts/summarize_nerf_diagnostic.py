"""Summarize every declared diagnostic run, including negative results."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from nerf_diagnostic_data import load_pixels


def aggregate(experiments):
    groups = {}
    for entry in experiments:
        config = entry["config"]
        key = f"{config['sampler']}_{config['count']}"
        groups.setdefault(key, []).append(entry)
    result = {}
    for key, values in groups.items():
        item = {"seed_count": len(values), "seeds": [value["config"]["seed"] for value in values]}
        metrics = ["mean_psnr", "mean_ssim"]
        if "mean_foreground_psnr" in values[0]:
            metrics.append("mean_foreground_psnr")
        for metric in metrics:
            numbers = [value[metric] for value in values]
            item[metric] = {"mean": float(np.mean(numbers)), "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
                            "per_seed": numbers}
        result[key] = item
    lookup = {(entry["config"]["count"], entry["config"]["seed"]): entry
              for entry in experiments if entry["config"]["sampler"] == "hierarchical"}
    differences = []
    for seed in (0, 1, 2):
        first, second = lookup[(1, seed)], lookup[(2, seed)]
        first_views = {view["id"]: view for view in first["views"]}
        second_views = {view["id"]: view for view in second["views"]}
        if first_views.keys() != second_views.keys():
            raise ValueError("Paired experiments must evaluate identical test views")
        differences.append({"seed": seed, "psnr": second["mean_psnr"] - first["mean_psnr"],
                            "ssim": second["mean_ssim"] - first["mean_ssim"],
                            "test_view_count": len(first_views),
                            "per_view_wins": {metric: sum(second_views[view_id][metric] > first_views[view_id][metric]
                                                          for view_id in first_views)
                                              for metric in ("psnr", "ssim")}})
    return {"groups": result, "paired_two_minus_one": differences,
            "two_better_on_both_metrics_all_seeds": all(item["psnr"] > 0 and item["ssim"] > 0 for item in differences)}


def comparison_sheet(folder, scene):
    frames = scene["test"][:4]
    scale = 2
    width, height = frames[0]["width"] * scale, frames[0]["height"] * scale
    row_height = height + 32
    board = Image.new("RGB", (width * 5, row_height * len(frames) + 24), "white")
    draw = ImageDraw.Draw(board)
    for column, label in enumerate(("Ground truth", "1 reference", "2 references", "4 references", "8 references")):
        draw.text((column * width + 6, 5), label + (" / seed 0" if column else ""), fill="black")
    for row, frame in enumerate(frames):
        pixels = (load_pixels(frame, scene["background"]) * 255).round().astype(np.uint8)
        image = Image.fromarray(pixels).resize((width, height), Image.Resampling.NEAREST)
        board.paste(image, (0, 24 + row * row_height))
        draw.text((6, 24 + row * row_height + height + 4), frame["id"], fill="black")
        for column, count in enumerate((1, 2, 4, 8), 1):
            run = folder / f"hierarchical_{count:02d}views_seed0/test"
            image = Image.open(run / f"{row:03d}.png").convert("RGB").resize((width, height), Image.Resampling.NEAREST)
            board.paste(image, (column * width, 24 + row * row_height))
            metric = json.loads((run / "metrics.json").read_text())["views"][row]
            draw.text((column * width + 6, 24 + row * row_height + height + 4),
                      f"PSNR {metric['psnr']:.2f} / SSIM {metric['ssim']:.3f}", fill="black")
    board.save(folder / "test_comparison.png")


def reference_sheet(folder, scene):
    frames = scene["training_order"]
    width, height = frames[0]["width"] * 2, frames[0]["height"] * 2
    board = Image.new("RGB", (width * 4, (height + 24) * 2), "white")
    draw = ImageDraw.Draw(board)
    for index, frame in enumerate(frames):
        pixels = (load_pixels(frame, scene["background"]) * 255).round().astype(np.uint8)
        image = Image.fromarray(pixels).resize((width, height), Image.Resampling.NEAREST)
        column, row = index % 4, index // 4
        board.paste(image, (column * width, row * (height + 24)))
        draw.text((column * width + 5, row * (height + 24) + height + 4),
                  f"Reference {index + 1}: {frame['id']}", fill="black")
    board.save(folder / "reference_inputs.png")


def plot_curves(root, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    counts = [1, 2, 4, 8]
    for axis, (name, summary) in zip(axes, summaries.items()):
        values = [summary["groups"][f"hierarchical_{count}"]["mean_psnr"] for count in counts]
        axis.errorbar(counts, [value["mean"] for value in values], yerr=[value["std"] for value in values],
                      marker="o", capsize=4, label="Hierarchical: mean +/- seed SD")
        if name == "lego":
            foreground = [summary["groups"][f"hierarchical_{count}"]["mean_foreground_psnr"] for count in counts]
            axis.plot(counts, [value["mean"] for value in foreground], marker="s", linestyle="--", label="Foreground only: mean")
        axis.set_title(f"{name.capitalize()} / fixed 1500 steps")
        axis.set_xlabel("Training reference images")
        axis.set_ylabel("Held-out mean PSNR (dB)")
        axis.set_xticks(counts)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(root / "view_count_psnr.png", dpi=180)
    figure.savefig(root / "view_count_psnr.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    lines = ["# Lego → LLFF Fern 稀疏视角 NeRF 诊断结果", "",
             "本实验复用现有 PyTorch 哈希网格 NeRF，排查输入视角数与射线采样。没有使用 ZeroNVS 教师、SDS、预测深度或新增表面约束；因此不能据此宣称 ZeroNVS 本身已经改进。", "",
             "## 固定协议", "",
             "每组 1500 步，每步 1024 条射线、每条 64 次场查询，哈希表 log2_size=17。学习率从 0.005 指数衰减到 0.0005。三种子主实验采用 32+32 分层重要性采样；均匀 64 点采样只做种子 0 的单/双图消融。单/双/四/八图使用嵌套参考集合、相同总射线预算、固定最终步；输入增加时单张图的射线份额减少。", "",
             "Lego 使用准确渲染相机，128×128，白底合成；官方 train 提供训练候选，val 每 10 张取 1 张共 10 张开发图，test 每 10 张取 1 张共 20 张测试图。另报告 alpha≥0.5 的前景 PSNR，alpha 不进入训练损失。", "",
             "Fern 使用官方 SfM 相机，128×96。按排序索引模 8：0 为测试、4 为开发、其余训练，共 15 张训练候选、2 张开发图、3 张测试图。归一化和近远边界由训练候选相机元数据确定。SfM 上游使用了场景多图，不能称为单/双图独立标定。", "",
             "两场景及实验矩阵在训练前写入 study.json/scene.json；仅完成全部 14 组后开放最终测试，并保存 test_seal.json 及模型摘要。没有依测试表现选图、选种子或挑检查点。前四个固定测试视角用于图示，未按效果挑选。", ""]
    for scene_name in ("lego", "fern"):
        folder = args.root / scene_name
        scene = json.loads((folder / "scene.json").read_text())
        report = json.loads((folder / "test_summary.json").read_text())
        summary = aggregate(report["experiments"])
        completed = [json.loads((folder / entry["run"] / "complete.json").read_text()) for entry in report["experiments"]]
        summary["total_training_seconds"] = sum(entry["training_seconds"] for entry in completed)
        summary["peak_allocated_gib"] = max(entry["peak_allocated_gib"] for entry in completed)
        summaries[scene_name] = summary
        comparison_sheet(folder, scene)
        reference_sheet(folder, scene)
        lines.extend([f"## {scene_name}", "", f"原始图像数：{scene['source_frame_count']}；第二参考相对旋转：{scene['second_reference_angle_deg']:.2f}°。", "",
                      "| 采样 | 输入图数 | 种子数 | 测试 PSNR 均值 ± 标准差 | 测试 SSIM 均值 ± 标准差 |",
                      "|---|---:|---:|---:|---:|"])
        for key, values in summary["groups"].items():
            sampler, count = key.split("_")
            formatted = []
            for metric in ("mean_psnr", "mean_ssim"):
                value = values[metric]
                formatted.append(f"{value['mean']:.4f}" + (f" ± {value['std']:.4f}" if value["std"] is not None else "（单种子）"))
            lines.append(f"| {sampler} | {count} | {values['seed_count']} | {formatted[0]} | {formatted[1]} |")
        lines.extend(["", "双图减单图的逐种子差值：", ""])
        for item in summary["paired_two_minus_one"]:
            lines.append(f"- seed {item['seed']}: PSNR {item['psnr']:+.4f} dB，SSIM {item['ssim']:+.5f}。")
            lines.append(f"  逐视角严格提升：PSNR {item['per_view_wins']['psnr']}/{item['test_view_count']}，"
                         f"SSIM {item['per_view_wins']['ssim']}/{item['test_view_count']}；平均提升不代表每个视角的所有指标都提升。")
        lines.extend(["", f"14 组累计纯训练用时 {summary['total_training_seconds']:.1f} 秒；峰值 PyTorch 已分配显存 {summary['peak_allocated_gib']:.3f} GiB。时间不包含下载、数据预处理、开发及测试渲染。", ""])
        if scene_name == "lego":
            lines.extend(["", "Lego 前景 PSNR（按图计算后对视角及种子等权平均）：", ""])
            for key, values in summary["groups"].items():
                lines.append(f"- {key}: {values['mean_foreground_psnr']['mean']:.4f} dB。")
        lines.extend(["", f"![固定测试视角对照]({scene_name}/test_comparison.png)", "",
                      "参考输入按顺序排列：单图取第 1 张，双图取前 2 张，四图取前 4 张，八图全部使用。", "",
                      f"![实际参考输入]({scene_name}/reference_inputs.png)", ""])
    lines.extend(["## 解释边界", "",
                  "上述结果只支持本次场景、输入选择、低分辨率和固定训练预算下的结论。没有复现原始 NeRF 或 ZeroNVS 论文训练设置。Lego 与 Fern 各一个场景，不能据此保证任意场景中双图必优于单图；三种子标准差不是跨场景置信区间。", "",
                  "单图不能唯一确定三维结构；更高训练 PSNR 不代表更高新视角质量。若 4/8 图改善而 2 图仍失败，应将问题定位到稀疏输入、覆盖和先验需求，而非把训练视角拟合当作成功。进一步验证 ZeroNVS 或表面约束需另行冻结教师、深度来源及等预算对照，不能混用本次纯 RGB 基线。", ""])
    (args.root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    plot_curves(args.root, summaries)
    (args.root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
