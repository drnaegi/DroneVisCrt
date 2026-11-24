"""Test plane-conditioned flow on fixed source pairs and target cameras."""
import argparse

import cv2
from PIL import Image

from relation_nvs.io import load_camera, sha256, write_json
from relation_nvs.stereo_transport import SurveyStereo, rectify_source_pair, unrectify_flow, qualify_flows
from tools.benchmark_stereo_transport import OUT, ROOT, raster, read, get_flow_model, estimate_flow


def run(groups):
    model = get_flow_model()
    for group in groups:
        folder = OUT/group
        records = {name:{} for name in ('rectified','dual_match')}
        for row in read(folder/'predictions.json'):
            cameras = [load_camera(c) for c in row['cameras']]
            images = [raster(ROOT/p, c) for p,c in zip(row['paths'][:2], cameras)]
            partners, homographies = rectify_source_pair(images,cameras[:2])
            residual = estimate_flow(model,images,partners)
            rectified = unrectify_flow(residual,cameras[:2],homographies)
            raw = estimate_flow(model,images)
            dual, chosen = qualify_flows(images,cameras[:2],(rectified,raw))
            for name, flow in [('rectified',rectified), ('dual_match',dual)]:
                atlas = SurveyStereo(images,cameras[:2],flow)
                rgb, _ = atlas.render(cameras[2])
                p = folder/row['case']/(name+'.png')
                Image.fromarray(rgb).save(p)
                records[name][row['case']] = dict(rgb_sha256=sha256(p))
            records['dual_match'][row['case']]['rectified_fraction'] = chosen
            print(group,row['case'],flush=True)
        for name, record in records.items():
            write_json(folder/(name+'.json'),record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='+')
    cv2.setNumThreads(2)
    run(parser.parse_args().groups)
