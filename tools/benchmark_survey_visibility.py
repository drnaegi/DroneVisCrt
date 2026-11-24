"""Disentangle image-guided depths, visible-surface rasterization and two-source blending."""
import argparse

import cv2
from PIL import Image
from relation_nvs.io import load_camera, sha256, write_json
from relation_nvs.survey_depth import load_moge,infer_depth
from relation_nvs.survey_flow import get_flow_model,estimate_flow
from relation_nvs.survey_visibility import render_surfaces
from relation_nvs.stereo_transport import SurveyStereo
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster


def run(groups):
    matcher,depth_model = get_flow_model(),load_moge()
    for group in groups:
        folder = OUT/group
        records = {}
        for row in read(folder/'predictions.json'):
            cameras = [load_camera(c) for c in row['cameras']]
            images = [raster(ROOT/p,c) for p,c in zip(row['paths'][:2],cameras)]
            mono = [infer_depth(depth_model,im,c) for im,c in zip(images,cameras)]
            flow = estimate_flow(matcher,images)
            for rule in ('qualified','median_stereo'):
                atlas = SurveyStereo(images,cameras[:2],flow,monocular=mono,prior_rule=rule)
                hard,blend,_ = render_surfaces(atlas,cameras[2])
                for mix,rgb in [('hard',hard),('blend',blend)]:
                    name = 'mesh_'+rule+'_'+mix
                    output = folder/row['case']/(name+'.png')
                    Image.fromarray(rgb).save(output)
                    records.setdefault(name,{})[row['case']] = dict(rgb_sha256=sha256(output))
            print(group,row['case'],flush=True)
        for name,rows in records.items():
            write_json(folder/(name+'.json'),rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='+')
    cv2.setNumThreads(2)
    run(parser.parse_args().groups)
