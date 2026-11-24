"""Test propagated uncertainty for depth fusion and sharp-source selection separately."""
import argparse

import numpy as np
from PIL import Image

from relation_nvs.io import load_camera, sha256, write_json
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_depth import infer_depth, load_moge
from relation_nvs.survey_flow import estimate_flow, get_flow_model
from relation_nvs.survey_visibility import combine_surface_and_inverse, render_surfaces
from tools.benchmark_stereo_transport import OUT, ROOT, read, raster


def run(groups):
    matcher, model = get_flow_model(), load_moge()
    for group in groups:
        folder = OUT/group
        outputs = {'precision_depth':{}, 'precision_donor':{}}
        for row in read(folder/'predictions.json'):
            cameras = [load_camera(c) for c in row['cameras']]
            images = [raster(ROOT/path,c) for path,c in zip(row['paths'][:2],cameras)]
            flow = estimate_flow(matcher,images)
            priors = [infer_depth(model,im,c) for im,c in zip(images,cameras)]
            renderer = SurveyStereo(images,cameras[:2],flow,monocular=priors,prior_rule='precision')
            target = cameras[-1]
            _, surface, _ = render_surfaces(renderer,target)
            for name in outputs:
                single, donor = renderer.render(target,precision_donor=(name=='precision_donor'))
                if name=='precision_depth':
                    original_donor = donor
                fused,_ = combine_surface_and_inverse(surface,single)
                path = folder/row['case']/(name+'.png')
                Image.fromarray(fused).save(path)
                outputs[name][row['case']] = dict(rgb_sha256=sha256(path), fits=renderer.prior_fits,
                        changed_fraction=float(np.mean(donor!=original_donor)))
            print(group,row['case'],flush=True)
        for name, mapping in outputs.items():
            write_json(folder/(name+'.json'),mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='+')
    run(parser.parse_args().groups)
