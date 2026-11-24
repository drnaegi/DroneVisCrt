"""Compare seven-parameter ray calibration and numerically matched monocular input."""
import argparse
import cv2
from PIL import Image

from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_depth import infer_depth,load_moge
from relation_nvs.survey_pointmap import load_pair_model,ordered_pointmaps,pointmap_precision,pointmap_renderer
from relation_nvs.survey_flow import get_flow_model,estimate_flow
from relation_nvs.survey_visibility import combine_surface_and_inverse,render_surfaces
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster,score

METHODS = ('pointmap_similarity','moge_matched_precision')


@pointmap_precision()
def run(groups):
    matcher,model,mono = get_flow_model(),load_pair_model(),load_moge()
    for group in groups:
        folder = OUT/group
        records = {name:{} for name in METHODS}
        for row in read(folder/'predictions.json'):
            cams = [load_camera(c) for c in row['cameras']]
            paths = [ROOT/p for p in row['paths'][:2]]
            if [sha256(p) for p in paths]!=row['source_hashes']:
                raise ValueError('Source bytes changed')
            images = [raster(p,c) for c,p in zip(cams,paths)]
            flow = estimate_flow(matcher,images)
            pair = ordered_pointmaps(model,images)
            sim,_ = pointmap_renderer(images,cams[:2],flow,pair,'similarity')
            depths=[infer_depth(mono,im,c) for im,c in zip(images,cams)]
            moge=SurveyStereo(images,cams[:2],flow,depths,'median_stereo')
            for name,atlas in zip(METHODS,(sim,moge)):
                inverse,_ = atlas.render(cams[2])
                _,blend,_ = render_surfaces(atlas,cams[2])
                rgb,_ = combine_surface_and_inverse(blend,inverse)
                path=folder/row['case']/(name+'.png')
                Image.fromarray(rgb).save(path)
                records[name][row['case']]=dict(rgb_sha256=sha256(path))
        for name,rows in records.items():
            write_json(folder/(name+'.json'),rows)
        score(group,only=METHODS,output='similarity_scores.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='+')
    cv2.setNumThreads(2)
    run(parser.parse_args().groups)
