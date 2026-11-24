"""Pin and download non-overlapping temporal triplets from two public aerial surveys."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path
import io
import re
import time

import requests
from PIL import Image

from relation_nvs.io import camera_dict, sha256, write_json
from floodnet_joint_data import camera

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'datasets'/'stereo_survey'
SURVEYS = [('drone_dataset_brighton_beach','9506a9bf09c9678b7d0b80b014af5b606c8a7255',3,1),
           ('odm_data_monterey_road_bend','597d22eaa3eadcb0d9e6f2620c93fdbeac0352c0',18,2)]


def fetch(url):
    for attempt in range(4):
        try:
            response=requests.get(url,timeout=(10,80))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt==3: raise
            time.sleep(2**attempt)


def extract(path):
    b=path.read_bytes()
    im=Image.open(io.BytesIO(b))
    exif=im.getexif()
    gps=exif.get_ifd(34853)
    def decimal(k):
        deg,m,s=map(float,gps[k])
        return (deg+m/60+s/3600)*(-1 if gps[k-1] in ('W','S') else 1)
    meta={k.decode():float(v) for k,v in re.findall(rb'drone-dji:(RelativeAltitude|GimbalYawDegree|GimbalPitchDegree|GimbalRollDegree)="([+\-0-9.]+)"',b[:262144])}
    return dict(lat=decimal(2),lon=decimal(4),width=im.width,height=im.height,
                focal=exif.get_ifd(34665)[41989],xmp=meta,frame=path.stem)


def main():
    for repo,revision,stride,radius in SURVEYS:
        base=DATA/repo
        base.mkdir(parents=True,exist_ok=True)
        api=f'https://api.github.com/repos/OpenDroneMap/{repo}/git/trees/{revision}?recursive=1'
        entries=sorted((i for i in fetch(api).json()['tree'] if i['path'].startswith('images/') and i['path'].lower().endswith('.jpg')),key=lambda i:i['path'])
        centers=list(range(radius,len(entries)-radius,stride))
        groups=[[i-radius,i+radius,i] for i in centers]
        ids=sorted({j for g in groups for j in g})
        def download(i):
            item=entries[i]
            path=base/f'{i:04d}.jpg'
            if not path.exists():
                url=f"https://raw.githubusercontent.com/OpenDroneMap/{repo}/{revision}/"+requests.utils.quote(item['path'])
                blob=fetch(url).content
                if sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()!=item['sha']:
                    raise ValueError('Git blob mismatch')
                tmp=path.with_suffix('.tmp')
                tmp.write_bytes(blob)
                tmp.replace(path)
            return i,extract(path)
        with ThreadPoolExecutor(8) as pool: metadata=dict(pool.map(download,ids))
        rows=[]
        for g in groups:
            heads=[metadata[i] for i in g]
            if any(x['xmp']['RelativeAltitude']<5 or x['xmp']['GimbalPitchDegree']>-68 for x in heads):continue
            cams=[]
            for x in heads:
                c=camera(x,heads[0])
                # The existing helper uses 512x384; rederive the equivalent focal at the actual aspect.
                import numpy as np
                from relation_nvs.geometry import Camera
                w,h=512,round(x['height']*512/x['width'])
                f=x['focal']*np.hypot(w,h)/np.hypot(36,24)
                cams.append(camera_dict(Camera(c.camera_id,[[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]],c.camera_to_world,w,h)))
            files=[f'{i:04d}.jpg' for i in g]
            rows.append(dict(files=files,cameras=cams,rolls=[x['xmp']['GimbalRollDegree'] for x in heads],sha256=[sha256(base/f) for f in files]))
        write_json(base/'manifest.json',dict(revision=revision,repo=repo,order='source1, source2, target',stride=stride,radius=radius,raw_paths={i:entries[i]['path'] for i in ids},rows=rows))
        print(repo,len(groups),len(rows),flush=True)


if __name__=='__main__':main()
