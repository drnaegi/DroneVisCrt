"""Pin and download non-overlapping temporal triplets from two public aerial surveys."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path
import io
import json
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

ADDITIONAL = {
    'sand_key': ('drone_dataset_sand_key','ab37a9680a8b663263f3af9e9af3131421b01f09',4,1),
    'cross': ('drone_dataset_sheffield_cross', '143f556ea3128835ab46cd694714f8f78319699b', 21, 1),
    'park1': ('drone_dataset_sheffield_park_1', '01e7a8dee3e2aa658f82598980770870bdf2fa82', 10, 1),
    'park3': ('drone_dataset_sheffield_park_3', 'c630b2d889ab744bfc17d848b8eb9d0999b67977', 5, 1),
    'noema': ('drone_dataset_noema', '635addace9afd01d4b7a7378483b9afcc842ee99', 16, 1),
}



def fetch(url, headers=None):
    for attempt in range(4):
        try:
            response=requests.get(url,headers=headers,timeout=(8,40))
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


def main(surveys):
    for repo,revision,stride,radius in surveys:
        base=DATA/repo
        base.mkdir(parents=True,exist_ok=True)
        api=f'https://api.github.com/repos/OpenDroneMap/{repo}/git/trees/{revision}?recursive=1'
        index_path = base / (revision + '.json')
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index['sha'] != revision or index['repo'] != repo:
                raise ValueError(f'Wrong Git index for {repo}')
            entries = index['images']
        else:
            entries = [i for i in fetch(api).json()['tree'] if (i['path'].startswith('images/') or '/' not in i['path']) and i['path'].lower().endswith('.jpg')]
            write_json(index_path, dict(sha=revision, repo=repo, images=entries))
        entries.sort(key=lambda i:i['path'])
        centers=list(range(radius,len(entries)-radius,stride))
        groups=[[i-radius,i+radius,i] for i in centers]
        ids=sorted({j for g in groups for j in g})
        def download(i):
            item=entries[i]
            path=base/f'{i:04d}.jpg'
            expected=item['sha']
            def verified(blob):
                return sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()==expected
            if not path.exists() or not verified(path.read_bytes()):
                url=f"https://raw.githubusercontent.com/OpenDroneMap/{repo}/{revision}/"+requests.utils.quote(item['path'])
                try:
                    response=requests.get(item['url'], headers={'Accept':'application/vnd.github.raw+json'},timeout=(5,20))
                    response.raise_for_status()
                    blob=response.content
                    if not verified(blob):
                        blob=fetch(url).content
                except requests.RequestException:
                    blob=fetch(url).content
                if not verified(blob):
                    raise ValueError('Git blob mismatch')
                tmp=path.with_suffix('.tmp')
                tmp.write_bytes(blob)
                tmp.replace(path)
            return i,extract(path)
        with ThreadPoolExecutor(8) as pool: metadata=dict(pool.map(download,ids))
        rows,excluded=[],[]
        for g in groups:
            heads=[metadata[i] for i in g]
            max_pitch = -45 if repo == 'drone_dataset_sand_key' else -68
            if any(x['xmp']['RelativeAltitude']<5 or x['xmp']['GimbalPitchDegree']>max_pitch for x in heads):
                excluded.append(dict(indices=g,reason='source or target outside altitude/pitch range',
                                     pitch_degrees=[x['xmp']['GimbalPitchDegree'] for x in heads],
                                     altitude_m=[x['xmp']['RelativeAltitude'] for x in heads]))
                continue
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
        write_json(base/'manifest.json',dict(revision=revision,repo=repo,order='source1, source2, target',stride=stride,radius=radius,raw_paths={i:entries[i]['path'] for i in ids},rows=rows,excluded=excluded))
        print(repo,len(groups),len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey', choices=list(ADDITIONAL))
    choice=parser.parse_args().survey
    main([ADDITIONAL[choice]] if choice else SURVEYS)
