import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/aerialmd_cpu'
sys.path.insert(0, str(ROOT / 'code/aerial-megadepth/mast3r'))
import mast3r.utils.path_to_dust3r
from dust3r.utils.image import load_images


def main():
    report = json.loads((RESULTS / 'dust3r_benchmark.json').read_text())
    loaded = load_images(report['images'], size=512, verbose=False)
    images = [(view['img'][0].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
              for view in reversed(loaded)]
    models = ['dust3r', 'mast3r']
    geometry = {}
    for model in models:
        with np.load(RESULTS / (model + '_geometry.npz')) as archive:
            geometry[model] = {key: archive[key] for key in archive.files}

    figure, axes = plt.subplots(2, 3, figsize=(15, 11), constrained_layout=True)
    figure.suptitle('AerialMegaDepth | Actual CPU inference outputs', fontsize=19)
    for row, image in enumerate(images):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title('Input B / reference camera' if row == 0 else 'Input A / second camera')
        axes[row, 0].axis('off')
    for column, model in enumerate(models, 1):
        depths = [geometry[model]['points_view%d' % view][..., 2] for view in [1, 2]]
        lower, upper = np.percentile(np.concatenate([depth.ravel() for depth in depths]), [2, 98])
        for row, depth in enumerate(depths):
            assert depth.shape == images[row].shape[:2]
            normalized = np.clip((depth - lower) / (upper - lower), 0, 1)
            plot = axes[row, column].imshow(normalized, cmap='magma', vmin=0, vmax=1)
            axes[row, column].set_title('Aerial-%s | reference-frame Z' % ('DUSt3R' if model == 'dust3r' else 'MASt3R'))
            axes[row, column].axis('off')
        figure.colorbar(plot, ax=list(axes[:, column]), shrink=0.55,
                        label='Normalized Z: dark = nearer, bright = farther')
    figure.supxlabel('Z is in camera B coordinates for BOTH rows. Each model uses its own 2–98% display range.\n'
                      'No ground truth or metric-scale comparison; these are geometry predictions, not generated RGB images.', fontsize=11)
    figure.savefig(RESULTS / 'geometry_comparison.png', dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(15, 11), constrained_layout=True)
    figure.suptitle('Prediction confidence | Same inputs and pixel alignment', fontsize=19)
    for row, image in enumerate(images):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title('Input B' if row == 0 else 'Input A')
        axes[row, 0].axis('off')
    for column, model in enumerate(models, 1):
        confidence = [geometry[model]['confidence_view%d' % view] for view in [1, 2]]
        log_values = [np.log1p(value) for value in confidence]
        lower, upper = np.percentile(np.concatenate([value.ravel() for value in log_values]), [2, 98])
        for row, value in enumerate(log_values):
            plot = axes[row, column].imshow(np.clip((value - lower) / (upper - lower), 0, 1),
                                           cmap='viridis', vmin=0, vmax=1)
            axes[row, column].set_title('Aerial-' + ('DUSt3R' if model == 'dust3r' else 'MASt3R'))
            axes[row, column].axis('off')
        figure.colorbar(plot, ax=list(axes[:, column]), shrink=0.55, label='Relative log-confidence')
    figure.supxlabel('Confidence uses log(1 + score), normalized per model to its 2–98% range.\n'
                      'Brighter pixels indicate higher model confidence, not calibrated accuracy probabilities.', fontsize=11)
    figure.savefig(RESULTS / 'confidence_comparison.png', dpi=150)
    plt.close(figure)

    figure = plt.figure(figsize=(14, 7), constrained_layout=True)
    figure.suptitle('Predicted colored point clouds | Unoptimized two-view output', fontsize=18)
    for column, model in enumerate(models, 1):
        axis = figure.add_subplot(1, 2, column, projection='3d')
        points_list, colors_list = [], []
        for view, image in enumerate(images, 1):
            points = geometry[model]['points_view%d' % view][::4, ::4].reshape(-1, 3)
            confidence = geometry[model]['confidence_view%d' % view][::4, ::4].ravel()
            colors = image[::4, ::4].reshape(-1, 3)
            valid = np.isfinite(points).all(axis=1) & (confidence >= np.median(confidence))
            points_list.append(points[valid])
            colors_list.append(colors[valid])
        points = np.concatenate(points_list)
        colors = np.concatenate(colors_list)
        points = points[:, [0, 2, 1]] * np.array([1, 1, -1])
        lower, upper = np.percentile(points, [1, 99], axis=0)
        visible = ((points >= lower) & (points <= upper)).all(axis=1)
        axis.scatter(*points[visible].T, c=colors[visible], s=1, depthshade=False, rasterized=True)
        center = (lower + upper) / 2
        radius = (upper - lower).max() / 2
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=25, azim=-65)
        axis.set_title('Aerial-' + ('DUSt3R' if model == 'dust3r' else 'MASt3R'))
        axis.set_xlabel('Camera B: X')
        axis.set_ylabel('Camera B: Z')
        axis.set_zlabel('Camera B: -Y')
    figure.supxlabel('Display only: stride 4, retain top 50% confidence per view, trim coordinate outliers (1–99%).\n'
                      'RGB comes from the input photos. Raw point predictions are not globally aligned or ground-truth validated.', fontsize=10)
    figure.savefig(RESULTS / 'pointcloud_comparison.png', dpi=160)
    plt.close(figure)
    print('Saved geometry_comparison.png, confidence_comparison.png, pointcloud_comparison.png')


if __name__ == '__main__':
    main()
