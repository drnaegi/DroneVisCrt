"""Compare metric ray shape recalibration against source-held-out shape selection."""
import argparse
import cv2
from PIL import Image

from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_flow import get_flow_model,estimate_flow
from relation_nvs.survey_pointmap import load_pair_model,ordered_pointmaps,calibrated_pair_depth,pointmap_precision
from relation_nvs.survey_visibility import render_surfaces,combine_surface_and_inverse
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster,score

RULES=('source_fit','rays_rescaled')


@pointmap_precision()
def run(groups):
    matcher,model=get_flow_model(),load_pair_model()
    for group in groups:
        folder=OUT/group
        records={r:{} for r in RULES}
        for row in read(folder/'predictions.json'):
            cams=[load_camera(c) for c in row['cameras']]
            paths=[ROOT/p for p in row['paths'][:2]]
            if [sha256(p) for p in paths]!=row['source_hashes']:
                raise ValueError('Source observations changed')
            images=[raster(p,c) for p,c in zip(paths,cams)]
            flow=estimate_flow(matcher,images)
            pair=ordered_pointmaps(model,images)
            depth,alignment=calibrated_pair_depth(pair,cams[:2],None,'rays')
            raw=[pair[j][0][j][...,2] for j in range(2)]
            for rule in RULES:
                atlas=SurveyStereo(images,cams[:2],flow,list(zip(raw,depth)),rule)
                inverse,_=atlas.render(cams[2])
                _,surface,_=render_surfaces(atlas,cams[2])
                rgb,_=combine_surface_and_inverse(surface,inverse)
                path=folder/row['case']/('pointmap_'+rule+'.png')
                Image.fromarray(rgb).save(path)
                records[rule][row['case']]=dict(rgb_sha256=sha256(path),fits=atlas.prior_fits,alignment=alignment)
            print(group,row['case'],flush=True)
        for rule in RULES:
            write_json(folder/('pointmap_'+rule+'.json'),records[rule])
        score(group,only=['pointmap_'+r for r in RULES],output='source_fit_scores.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups',nargs='+')
    cv2.setNumThreads(2)
    run(p.parse_args().groups)
