"""Compare frozen stereo pointmap priors in the same camera and rendering pipeline."""
import argparse

import cv2
import numpy as np
from PIL import Image

from relation_nvs.io import load_camera, sha256, write_json
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_flow import estimate_flow, get_flow_model
from relation_nvs.survey_pointmap import (PUBLISHED_SHA, default_checkpoint, load_pair_model,
                                         ordered_pointmaps, pointmap_renderer)
from relation_nvs.survey_visibility import combine_surface_and_inverse, render_surfaces
from tools.benchmark_stereo_transport import OUT, ROOT, PAIR_METHODS, raster, read, score

DEPTH_CONTROLS = ('detail_median_stereo_fixed', 'pointmap_global_optimizer')
DIAGNOSTIC = 'syndrone_heights'


def methods_for(group):
    if group == DIAGNOSTIC:
        return PAIR_METHODS + DEPTH_CONTROLS
    if group == 'syndrone_transfer':
        return PAIR_METHODS + DEPTH_CONTROLS[:1]
    return PAIR_METHODS


def optimized_depth(row, cameras):
    town = row['case'].split('_')[0]
    folder = ROOT/'outputs/relation_nvs_20261120/calibrated_dust3r_two_source_v2'
    protocol = read(folder/'protocol_before_prediction.json')
    refs = next(r['references'] for r in protocol['cases'] if r['town']==town)
    if [r['rgb_sha256'] for r in refs] != row['source_hashes']:
        raise ValueError('Optimizer inputs differ')
    path = folder/town/'prediction.npz'
    digest = sha256(path)
    if digest != read(path.parent/'status.json')['prediction_sha256']:
        raise ValueError('Optimizer prediction changed')
    with np.load(path) as values:
        if not (np.allclose(values['intrinsics'],[c.intrinsics for c in cameras],atol=.001) and
                np.allclose(values['poses'],[c.camera_to_world for c in cameras],atol=.001)):
            raise ValueError('Optimizer cameras differ')
        return values['depth'], digest


from relation_nvs.survey_pointmap import pointmap_precision


@pointmap_precision()
def generate(groups):
    from relation_nvs.survey_depth import infer_depth, load_moge
    if sha256(default_checkpoint()) != PUBLISHED_SHA:
        raise ValueError('Unexpected DUSt3R weights')
    matcher, network = get_flow_model(), load_pair_model()
    mono_model = load_moge() if any(DEPTH_CONTROLS[0] in methods_for(g) for g in groups) else None
    for group in groups:
        folder = OUT/group
        diagnostic = group == DIAGNOSTIC
        methods = methods_for(group)
        records = {name:{} for name in methods}
        for row in read(folder/'predictions.json'):
            cams = [load_camera(c) for c in row['cameras']]
            sources = [ROOT / p for p in row['paths'][:2]]
            if [sha256(p) for p in sources] != row['source_hashes']:
                raise ValueError('Source observations have changed')
            images = [raster(p,c) for p,c in zip(sources,cams)]
            flow = estimate_flow(matcher,images)
            pair = ordered_pointmaps(network,images)
            for name in methods:
                record = {}
                if name in PAIR_METHODS:
                    atlas, record['alignment'] = pointmap_renderer(images,cams[:2],flow,pair,name.removeprefix('pointmap_'))
                elif name == 'pointmap_global_optimizer':
                    depths, record['upstream_depth_sha256'] = optimized_depth(row,cams[:2])
                    atlas = SurveyStereo(images,cams[:2],flow,depths,'metric_pair')
                else:
                    depths = [infer_depth(mono_model,im,c) for im,c in zip(images,cams)]
                    atlas = SurveyStereo(images,cams[:2],flow,depths,'median_stereo')
                case_folder = folder / row['case']
                if diagnostic:
                    axial = [(geometry[1]-c.camera_to_world[:3,3])@c.camera_to_world[:3,2]
                             for c,geometry in zip(cams,atlas.geometry)]
                    depth_path = case_folder / (name+'_depth.npz')
                    np.savez_compressed(depth_path,depth=np.asarray(axial,np.float32))
                    record['depth_sha256'] = sha256(depth_path)
                inverse,_ = atlas.render(cams[2])
                _,surface,_ = render_surfaces(atlas,cams[2])
                rgb,_ = combine_surface_and_inverse(surface,inverse)
                path = case_folder / (name+'.png')
                Image.fromarray(rgb).save(path)
                record['rgb_sha256'] = sha256(path)
                records[name][row['case']] = record
            print(group,row['case'],flush=True)
        for name,values in records.items():
            write_json(folder/(name+'.json'),values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generate','score'])
    parser.add_argument('groups',nargs='+')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.stage == 'generate':
        generate(args.groups)
    else:
        for group in args.groups:
            methods = methods_for(group)
            score(group, only=methods, output='pointmap_scores.json')
