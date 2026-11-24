"""Prepare a camera-only static-object transfer split from the official NeRF Lego poses."""
from pathlib import Path

import numpy as np

from relation_nvs.geometry import Camera
from relation_nvs.aerial_mosaic import plane_projection
from relation_nvs.io import camera_dict,write_json
from tools.benchmark_stereo_transport import ROOT,OUT,read


def prepare():
    directory=ROOT/'datasets/nerf_diagnostic/nerf_synthetic/lego'
    def cameras(split):
        data=read(directory/f'transforms_{split}.json')
        f=256/np.tan(data['camera_angle_x']/2)
        for i,frame in enumerate(data['frames']):
            if split=='test' and i%8:continue
            pose=np.array(frame['transform_matrix'])@np.diag([1,-1,-1,1])
            if -pose[2,2]<=np.sin(data['camera_angle_x']/2 + np.deg2rad(12)):continue
            camera=Camera(f'{split}_{i:03d}',[[f,0,255.5],[0,f,255.5],[0,0,1]],pose,512,512)
            plane_projection(camera)
            yield directory/(frame['file_path']+'.png'),camera
    train=list(cameras('train'))
    rows=[]
    for p,c in cameras('test'):
        sources=sorted(train,key=lambda pair:np.linalg.norm(pair[1].camera_to_world[:3,3]-c.camera_to_world[:3,3]))[:2]
        rr=sources+[(p,c)]
        rows.append(dict(paths=[str(f.relative_to(ROOT)) for f,_ in rr],cameras=[camera_dict(v) for _,v in rr]))
    write_json(OUT/'lego_cases.json',rows)


if __name__=='__main__':prepare()
