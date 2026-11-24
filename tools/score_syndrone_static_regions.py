"""Independent simulator visibility and static labels applied after all RGB predictions."""
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.syndrone import decode_depth
from tools.acquire_syndrone_height_transfer import DATA,HEIGHTS
from tools.audit_survey_pointmap_depth import depth_masks
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster

STATIC_IDS = [1,2,5,6,7,8,9,11,12,14,15,16,17,18,19,22]


def score():
    folder = OUT/'syndrone_transfer'
    scores = {}
    for file in ('scores.json','rematch_scores.json','global_optimizer_scores.json','similarity_scores.json','mapanything_scores.json','mapanything_scale_scores.json'):
        if not (folder/file).is_file():
            continue
        for row in read(folder/file)['rows']:
            scores[row['case'],row['method']] = row
    rows = []
    for pred in read(folder/'predictions.json'):
        paths = [ROOT/p for p in pred['paths']]
        town,frame,_ = paths[0].stem.rsplit('_',2)
        cams = [load_camera(c) for c in pred['cameras']]
        evaluation = DATA/town/'evaluation'
        manifest = read(evaluation/'members.json')
        def truth(h,kind):
            path = evaluation/f'{frame}_{h}_{kind}.png'
            if sha256(path)!=manifest[path.name]['sha256']:
                raise ValueError('Evaluation truth changed')
            with Image.open(path) as im:
                return np.array(im)
        size = (cams[2].width,cams[2].height)
        depths = [np.asarray(Image.fromarray(decode_depth(truth(h,'depth'))).resize(size,Image.Resampling.NEAREST))
                  for h in HEIGHTS]
        mask = np.isin(truth(50,'semantic'),STATIC_IDS).astype(np.float32)
        mask = np.asarray(Image.fromarray(mask).resize(size,Image.Resampling.BOX))>=1.
        mask &= np.isfinite(depths[2]) & (depths[2]>0)
        n_visible = np.sum([depth_masks(depths[2],depths[j],cams[2],cams[j])['peer_visible'] for j in (0,1)],axis=0)
        regions = dict(static=mask,unseen=mask&(n_visible==0),one=mask&(n_visible==1),both=mask&(n_visible==2))
        windows = {k:binary_erosion(v,np.ones((11,11),bool)) for k,v in regions.items()}
        target = raster(paths[2],cams[2])
        digest = sha256(paths[2])
        for (case,method),ev in scores.items():
            if case!=pred['case']:
                continue
            if ev['target_sha256']!=digest:
                raise ValueError('RGB evaluation target differs')
            path = folder/case/(method+'.png')
            artifacts = pred['artifacts']
            expected = artifacts[method] if method in artifacts else read(folder/(method+'.json'))[case]['rgb_sha256']
            if sha256(path)!=expected:
                raise ValueError('Prediction changed')
            with Image.open(path) as im:
                pred_rgb = np.asarray(im)
            error = np.mean((target.astype(float)-pred_rgb)**2,axis=-1)
            _,field = structural_similarity(target,pred_rgb,channel_axis=-1,full=True,gaussian_weights=True,
                                           sigma=1.5,use_sample_covariance=False,data_range=255)
            for region,m in regions.items():
                window = windows[region]
                rows.append(dict(case=case,town=town,region=region,method=method,pixels=int(m.sum()),static_fraction=m.mean(),
                                 windows=int(window.sum()),psnr=float(-10*np.log10(max(error[m].mean()/255**2,1e-12))) if m.any() else None,
                                 ssim=float(field[window].mean()) if window.any() else None,full_lpips=ev['lpips']))
    means={}
    for r in rows:
        means.setdefault(r['method'],{})[r['region']]={}
    for method,regions in means.items():
        for region in regions:
            values=[r for r in rows if r['method']==method and r['region']==region]
            regions[region]={k:float(np.mean([r[k] for r in values if r[k] is not None])) if any(r[k] is not None for r in values) else None
                             for k in ('psnr','ssim','full_lpips','static_fraction')}
            regions[region]['targets_with_ssim']=sum(r['ssim'] is not None for r in values)
    write_json(folder/'static_region_scores.json',dict(rows=rows,means=means))


if __name__=='__main__':
    cv2.setNumThreads(2)
    score()
