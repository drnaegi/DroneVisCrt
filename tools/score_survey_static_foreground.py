"""Evaluate Lego's independently known opaque foreground and fully valid SSIM windows."""
import json

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from tools.benchmark_stereo_transport import OUT,ROOT,read,raster
from relation_nvs.io import load_camera,sha256,write_json


def main():
    folder=OUT/'lego'
    manifest=read(folder/'predictions.json')
    group_scores=read(folder/'scores.json')
    records={(r['case'],r['method']):r for r in group_scores['rows']}
    rows=[]
    for r in manifest:
        camera=load_camera(r['cameras'][2])
        target_path=ROOT/r['paths'][2]
        truth=raster(target_path,camera)
        with Image.open(target_path) as im:
            alpha=np.array(im.getchannel('A').resize((camera.width,camera.height),Image.Resampling.BOX))
        mask=alpha>=253
        valid=binary_erosion(mask,structure=np.ones((11,11),bool))
        if not valid.any():raise ValueError('Empty static support')
        for method in group_scores['mean']:
            if records[r['case'],method]['target_sha256']!=sha256(target_path):raise ValueError('Target changed')
            with Image.open(folder/r['case']/(method+'.png')) as im:
                prediction=np.asarray(im)
            error=np.mean((truth[mask].astype(float)-prediction[mask])**2)
            _,field=structural_similarity(truth,prediction,channel_axis=-1,full=True,gaussian_weights=True,
                                         sigma=1.5,use_sample_covariance=False,data_range=255)
            rows.append(dict(case=r['case'],method=method,pixels=int(mask.sum()),windows=int(valid.sum()),
                             foreground_psnr=float(-10*np.log10(max(error/255**2,1e-12))),
                             foreground_ssim=float(field[valid].mean()),static_lpips=records[r['case'],method]['lpips']))
    means={k:{m:float(np.mean([r[m] for r in rows if r['method']==k]))
              for m in ['foreground_psnr','foreground_ssim','static_lpips']} for k in group_scores['mean']}
    write_json(folder/'foreground_scores.json',dict(rows=rows,mean=means))


if __name__=='__main__':main()
