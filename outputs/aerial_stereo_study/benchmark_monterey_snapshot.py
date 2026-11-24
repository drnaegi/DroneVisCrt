"""Fixed two-input NVS controls on survey cameras. Generation never loads target RGB."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from relation_nvs.io import load_camera, camera_dict, sha256, write_json
from relation_nvs.aerial_mosaic import nadir_camera, SurveyCalibration
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_flow import get_flow_model, estimate_flow

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/aerial_stereo_study'
MODES=('plane','flow_plane','stereo','transport')


def read(p):
    return json.loads(p.read_text())


def cases(group):
    if group in ('brighton','monterey'):
        sub={'brighton':'drone_dataset_brighton_beach','monterey':'odm_data_monterey_road_bend'}[group]
        path=ROOT/'datasets/stereo_survey'/sub/'manifest.json'
        for r in read(path)['rows']:
            yield [path.parent/f for f in r['files']], [nadir_camera(load_camera(c),v) for c,v in zip(r['cameras'],r['rolls'])]
    elif group=='matrixcity':
        path=ROOT/'datasets/matrixcity_fusion_20261116'
        from relation_nvs.geometry import Camera
        def frames(split):
            for frame in read(path/'pose'/f'transforms_{split}.json')['frames']:
                name=frame['file_path'].removeprefix('../')
                if not name.startswith('aerial/'):continue
                m=np.asarray(frame['transform_matrix'])
                m[:3,1:3]*=-1
                m[:3,3]*=100
                m[3]=[0,0,0,1]
                k=[[frame['fl_x'],0,frame['cx']],[0,frame['fl_y'],frame['cy']],[0,0,1]]
                c=Camera(Path(name).stem,k,m,frame['w'],frame['h']).resized(512,288)
                yield path/name,c
        refs=list(frames('train'))
        for p,c in frames('test'):
            # Official pose only, nearest neighbours in centre + optical-axis chord distance.
            near=sorted(refs,key=lambda x:np.linalg.norm(x[1].camera_to_world[:3,3]-c.camera_to_world[:3,3])+100*np.linalg.norm(x[1].camera_to_world[:3,2]-c.camera_to_world[:3,2]))
            rows=near[:2]+[(p,c)]
            yield [x[0] for x in rows],[x[1] for x in rows]
    else:
        path=ROOT/'outputs/aerial_mosaic_validation/calibrated'/group/'sites.json'
        for s in read(path)['sites']:
            for t in sorted({t for pair in s['options'] for t in pair}):
                ids=[s['source'],s['helper'],t]
                yield [ROOT/'datasets/floodnet_joint/images'/f'{i}.jpg' for i in ids],[load_camera(s['cameras'][str(i)]) for i in ids]


def raster(path,c):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((c.width,c.height),Image.Resampling.LANCZOS))


def generate(group):
    dest=OUT/group
    dest.mkdir(parents=True,exist_ok=True)
    model=get_flow_model()
    rows=[]
    for paths,cams in cases(group):
        images=[raster(p,c) for p,c in zip(paths[:2],cams[:2])]
        if group in ('brighton','monterey'):
            fit=SurveyCalibration.estimate(images,cams[:2])
            cams=[fit.apply(c) for c in cams]
        key='_'.join(p.stem for p in paths)
        folder=dest/key
        folder.mkdir(exist_ok=True)
        record=dict(paths=[str(p.relative_to(ROOT)) for p in paths],cameras=[camera_dict(c) for c in cams],
                    source_hashes=[sha256(p) for p in paths[:2]],group=group,case=key)
        atlas=SurveyStereo(images,cams[:2],estimate_flow(model,images))
        for mode in MODES:
            pred,prov=atlas.render(cams[2],mode)
            Image.fromarray(pred).save(folder/f'{mode}.png')
            if mode=='plane':
                Image.fromarray(prov).save(folder/'support.png')
        record['artifacts']={m:sha256(folder/f'{m}.png') for m in MODES}
        rows.append(record)
        print(group,key,flush=True)
    write_json(dest/'predictions.json',rows)


def score(group):
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    import torch,lpips
    metric=lpips.LPIPS(net='alex').cuda().eval()
    def tensor(im):
        return torch.tensor(im.copy()).cuda().permute(2,0,1)[None].float()/127.5-1
    dest=OUT/group
    rows=[]
    extras = {name: read(dest/f'{name}.json') for name in ('depth_prior', 'mvsplat_acid_native_plane_bg', 'rectified', 'dual_match', 'altitude', 'median_plane', 'median_stereo')
              if (dest/f'{name}.json').is_file()}
    for r in read(dest/'predictions.json'):
        cam=load_camera(r['cameras'][-1])
        truth=ROOT/r['paths'][-1]
        target=raster(truth,cam)
        folder=dest/r['case']
        support=np.asarray(Image.open(folder/'support.png'))>0
        artifacts = {**r['artifacts'], **{k:v[r['case']]['rgb_sha256'] for k,v in extras.items()}}
        for m,expected in artifacts.items():
            p=folder/f'{m}.png'
            assert sha256(p)==expected
            im=np.asarray(Image.open(p))
            diff=(im.astype(float)-target)**2
            def psnr(mask):
                return float(-10*np.log10(max(diff[mask].mean()/255**2,1e-12))) if mask.any() else None
            with torch.inference_mode():
                rows.append(dict(case=r['case'],source=r['paths'][0],target=truth.name,target_sha256=sha256(truth),method=m,
                        psnr=peak_signal_noise_ratio(target,im,data_range=255),ssim=structural_similarity(target,im,channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,data_range=255),
                        lpips=metric(tensor(target),tensor(im)).item(),observed_psnr=psnr(support),complement_psnr=psnr(~support),plane_support=float(support.mean())))
    means={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in ('lpips','psnr','ssim')} for m in sorted({r['method'] for r in rows})}
    write_json(dest/'scores.json',dict(rows=rows,mean=means))
    print(group,means,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['generate','score'])
    p.add_argument('groups',nargs='+')
    a=p.parse_args()
    cv2.setNumThreads(2)
    for g in a.groups:globals()[a.stage](g)
