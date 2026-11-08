import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dust3r', 'mast3r'], required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--images', nargs=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/aerialmd_cpu')
    args = parser.parse_args()
    if args.threads < 1 or args.runs < 1:
        parser.error('threads and runs must be positive')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    sys.path.insert(0, str(ROOT / 'code/aerial-megadepth/mast3r'))

    import numpy as np
    import torch
    import mast3r.utils.path_to_dust3r
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from mast3r.model import AsymmetricMASt3R

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model_class = AsymmetricCroCo3DStereo if args.model == 'dust3r' else AsymmetricMASt3R
    checkpoint = ROOT / 'checkpoints/aerial-megadepth' / ('checkpoint-aerial-' + args.model)
    started = time.perf_counter()
    model = model_class.from_pretrained(str(checkpoint), local_files_only=True, strict=True)
    model = model.to('cpu').eval()
    load_seconds = time.perf_counter() - started
    print('Model loaded in %.3f seconds' % load_seconds, flush=True)
    paths = args.images or [
        str(ROOT / 'code/aerial-megadepth/assets/siteACC0003-finearts' / filename)
        for filename in [
            'siteACC0003-camA005-2023-10-16-13-32-00-000181.jpg',
            'siteACC0003-camA010-2023-12-21-19-43-25-000309.jpg',
        ]
    ]
    started = time.perf_counter()
    images = load_images(paths, size=args.size)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    preprocess_seconds = time.perf_counter() - started
    times = []
    for iteration in range(args.runs + 1):
        started = time.perf_counter()
        with torch.inference_mode():
            output = inference(pairs, model, 'cpu', batch_size=1, verbose=False)
        elapsed = time.perf_counter() - started
        if iteration == 0:
            warmup_seconds = elapsed
            print('First inference: %.3f seconds' % elapsed, flush=True)
        else:
            times.append(elapsed)
            print('Measured inference %d: %.3f seconds' % (iteration, elapsed), flush=True)
    prediction_summary = {}
    for view in ['pred1', 'pred2']:
        prediction_summary[view] = {}
        for name, value in output[view].items():
            tensors = value if isinstance(value, list) else [value]
            if not tensors or not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
                raise TypeError('Unexpected prediction: ' + view + '.' + name)
            finite = all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
            prediction_summary[view][name] = {
                'shapes': [list(tensor.shape) for tensor in tensors], 'finite': finite,
            }
            if not finite:
                raise ValueError('Nonfinite prediction: ' + view + '.' + name)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / (args.model + '_geometry.npz'),
        points_view1=output['pred1']['pts3d'][0].numpy(),
        points_view2=output['pred2']['pts3d_in_other_view'][0].numpy(),
        confidence_view1=output['pred1']['conf'][0].numpy(),
        confidence_view2=output['pred2']['conf'][0].numpy(),
    )
    result = {
        'model': args.model,
        'checkpoint': str(checkpoint),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'device': 'cpu',
        'dtype': str(next(model.parameters()).dtype),
        'threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'images': paths,
        'input_shapes': [list(view['img'].shape) for view in images],
        'directed_pairs': len(pairs),
        'batch_size': 1,
        'load_seconds': load_seconds,
        'preprocess_seconds': preprocess_seconds,
        'first_inference_seconds': warmup_seconds,
        'measured_seconds': times,
        'mean_seconds': statistics.mean(times),
        'median_seconds': statistics.median(times),
        'peak_process_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
        'predictions': prediction_summary,
        'timing_scope': 'Official inference on two directed pairs (A,B and B,A); excludes model loading, image preprocessing, matching, pose estimation, global alignment, and output serialization.',
    }
    report = args.output / (args.model + '_benchmark.json')
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
