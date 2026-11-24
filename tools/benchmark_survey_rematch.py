"""Measure image and plane prealignment against true 3D surface prealignment."""
import argparse
import cv2
from PIL import Image

from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_pointmap import calibrated_pair_depth, ordered_pointmaps, load_pair_model, rematch_candidates
from relation_nvs.survey_flow import get_flow_model, estimate_flow
from relation_nvs.survey_visibility import render_surfaces,combine_surface_and_inverse
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster,score

METHODS = ('surface_direct','surface_refined','ground_selected','surface_selected')


from relation_nvs.survey_pointmap import pointmap_precision


@pointmap_precision()
def generate(groups):
    model, matcher = load_pair_model(),get_flow_model()
    for group in groups:
        folder = OUT/group
        records = {m:{} for m in METHODS}
        for row in read(folder/'predictions.json'):
            cams = [load_camera(c) for c in row['cameras']]
            sources = [ROOT/p for p in row['paths'][:2]]
            if [sha256(p) for p in sources]!=row['source_hashes']:
                raise ValueError('Source file changed')
            images = [raster(p,c) for c,p in zip(cams,sources)]
            flows = estimate_flow(matcher,images)
            depths,_ = calibrated_pair_depth(ordered_pointmaps(model,images),cams[:2],flows,'rays')
            candidates = rematch_candidates(images,cams[:2],depths,flows,matcher)
            for method,flow in candidates.items():
                atlas = SurveyStereo(images,cams[:2],flow,depths,'metric_pair')
                inverse,_ = atlas.render(cams[2])
                _,blend,_ = render_surfaces(atlas,cams[2])
                pred,_ = combine_surface_and_inverse(blend,inverse)
                path = folder/row['case']/(method+'.png')
                Image.fromarray(pred).save(path)
                records[method][row['case']] = dict(rgb_sha256=sha256(path))
            print(group,row['case'],flush=True)
        for m,values in records.items():
            write_json(folder/(m+'.json'),values)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['generate','score'])
    p.add_argument('groups',nargs='+')
    a=p.parse_args()
    cv2.setNumThreads(2)
    if a.stage=='generate':
        generate(a.groups)
    else:
        for g in a.groups:
            score(g,only=METHODS,output='rematch_scores.json')
