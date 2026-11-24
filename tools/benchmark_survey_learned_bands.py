"""Fit band weights on disjoint regions, seal them, then render the held-out flights."""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from relation_nvs.io import load_camera, sha256, write_json
from relation_nvs.survey_bands import FEATURE_COUNTS, band_inputs, combine, decompose
from tools.benchmark_stereo_transport import OUT, ROOT, raster, read
from tools.summarize_survey_nvs import GEOGRAPHY


def examples(group):
    folder = OUT/group
    sources = ['mesh_qualified_blend','depth_prior']
    hashes = [read(folder/(m+'.json')) for m in sources]
    for row in read(folder/'predictions.json'):
        cameras = [load_camera(c) for c in row['cameras']]
        images = []
        for m,refs in zip(sources,hashes):
            path = folder/row['case']/(m+'.png')
            if sha256(path) != refs[row['case']]['rgb_sha256']:
                raise ValueError(f'Changed rendering {path}')
            with Image.open(path) as im:
                images.append(np.array(im.convert('RGB')))
        yield row, cameras, images


def sufficient_statistics(group):
    rows, targets = [], {}
    for row,cs,images in examples(group):
        path = ROOT/row['paths'][-1]
        targets[row['case']] = sha256(path)
        target_bands = decompose(raster(path,cs[-1]).astype(np.float32)/255)
        bands = []
        for (base,delta,features),target in zip(band_inputs(*images,cs),target_bands,strict=True):
            # Every fourth pixel in both axes; equal frame/flight/region weights.
            f = features[::4,::4].reshape(-1,features.shape[-1]).astype(np.float64)
            d = delta[::4,::4].reshape(-1,3).astype(np.float64)
            y = (target-base)[::4,::4].reshape(-1,3)
            gram = (f.T*np.sum(d*d,axis=1))@f/len(f)
            rhs = f.T@np.sum(d*y,axis=1)/len(f)
            bands.append(np.column_stack((gram,rhs)))
        rows.append(bands)
    return np.mean(rows,axis=0),targets


def fit(stats):
    weights = {}
    for name,n in FEATURE_COUNTS.items():
        bands = []
        for block in stats:
            gram, rhs = block[:n,:n],block[:n,-1]
            penalty = np.eye(n)*max(1e-12,gram[0,0]*.001)
            # The unregularized intercept fits the best global color mix.
            penalty[0,0] = 0
            bands.append(np.linalg.lstsq(gram+penalty,rhs,rcond=None)[0].tolist())
        weights[name] = bands
    return weights


def train():
    regions = defaultdict(list)
    hashes = {}
    for group,region in GEOGRAPHY.items():
        block,hashes[group] = sufficient_statistics(group)
        regions[region].append(block)
    train_region = {k:np.mean(v,axis=0) for k,v in regions.items()}
    for held_out in [*regions,'deployment']:
        inputs = [k for k in regions if k!=held_out]
        groups = [g for g,r in GEOGRAPHY.items() if r in inputs]
        bundle = dict(held_out=held_out,training_regions=inputs,training_groups=groups,
                      training_targets={g:hashes[g] for g in groups},
                      weights=fit(np.mean([train_region[k] for k in inputs],axis=0)))
        write_json(OUT/'band_weights'/(held_out+'.json'),bundle)


def apply(group):
    region = GEOGRAPHY.get(group,'deployment')
    bundle_path = OUT/'band_weights'/(region+'.json')
    bundle = read(bundle_path)
    if region in bundle['training_regions'] or group in bundle['training_groups']:
        raise ValueError(f'Test region overlaps calibration for {group}')
    records = defaultdict(dict)
    for row,cs,images in examples(group):
        for variant,coeff in bundle['weights'].items():
            name = 'learned_bands_'+variant
            rgb = combine(*images,cs,coeff)
            path = OUT/group/row['case']/(name+'.png')
            Image.fromarray(rgb).save(path)
            records[name][row['case']] = dict(rgb_sha256=sha256(path),weights_sha256=sha256(bundle_path))
    for name,rows in records.items():
        write_json(OUT/group/(name+'.json'),rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','apply'])
    parser.add_argument('groups',nargs='*')
    args = parser.parse_args()
    if args.stage=='train':
        train()
    else:
        for group in args.groups:
            apply(group)
