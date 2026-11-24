"""Estimate depth from the two source RGBs and test source-qualified monocular transfer."""
import argparse

import cv2
from PIL import Image

from relation_nvs.io import load_camera, sha256, write_json
from relation_nvs.stereo_transport import SurveyStereo
from tools.benchmark_stereo_transport import OUT, ROOT, raster, read, get_flow_model, estimate_flow


from relation_nvs.survey_depth import load_moge, infer_depth


def run(groups, rules):
    flow_model, depth_model = get_flow_model(), load_moge()
    for group in groups:
        folder = OUT/group
        manifest = read(folder/'predictions.json')
        results = {name:{} for name in rules}
        for row in manifest:
            cameras = [load_camera(c) for c in row['cameras']]
            images = [raster(ROOT/p, c) for p,c in zip(row['paths'][:2], cameras)]
            depth = [infer_depth(depth_model, im, c) for im,c in zip(images, cameras)]
            flow = estimate_flow(flow_model, images)
            for name in rules:
                rule = 'qualified' if name == 'depth_prior' else name
                atlas = SurveyStereo(images, cameras[:2], flow, monocular=depth, prior_rule=rule)
                pred, _ = atlas.render(cameras[2])
                output = folder/row['case']/(name+'.png')
                Image.fromarray(pred).save(output)
                results[name][row['case']] = dict(fits=atlas.prior_fits, rgb_sha256=sha256(output))
            print(group, row['case'], flush=True)
        for name, records in results.items():
            write_json(folder/(name+'.json'), records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups', nargs='+')
    parser.add_argument('--rules', nargs='+', choices=['depth_prior','altitude','median_plane','median_stereo'],
                        default=['depth_prior','altitude','median_plane','median_stereo'])
    cv2.setNumThreads(2)
    args = parser.parse_args()
    run(args.groups,args.rules)
