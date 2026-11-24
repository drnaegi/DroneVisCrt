"""Freeze geographically separated sites and the camera/annotation budget before generating views."""
import argparse
import itertools
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from acquire_floodnet_supervised_pairs import distance, read, signed_url, safe_member, RangeArchive
from index_floodnet_joint import CACHE, DEV, OUT
from evaluate_floodnet_grass_strips import W,H
from relation_nvs.io import sha256, write_json, camera_dict
from relation_nvs.geometry import Camera


def labels_path(i):
    p=CACHE/'masks'/f'{i}_lab.png'
    return p if p.exists() else DEV/'masks'/p.name


def labels(i):
    with Image.open(labels_path(i)) as im:
        raw=np.array(im.resize((W,H),Image.Resampling.NEAREST))
    return np.select([np.isin(raw,[3,4]), raw==5, raw==9],[1,2,3],0).astype('uint8')


def camera(a,origin):
    xmp=a['xmp']
    yaw,pitch,roll=np.radians([xmp['Gimbal'+k+'Degree'] for k in ('Yaw','Pitch','Roll')])
    east=np.array([np.cos(yaw),-np.sin(yaw),0.])
    fwd=np.array([np.sin(yaw)*np.cos(pitch),np.cos(yaw)*np.cos(pitch),np.sin(pitch)])
    down=np.cross(fwd,east)
    pose=np.eye(4)
    pose[:3,:3]=np.column_stack((east*np.cos(roll)+down*np.sin(roll), down*np.cos(roll)-east*np.sin(roll),fwd))
    dy,dx=np.radians([a['lat']-origin['lat'],a['lon']-origin['lon']])*6378137
    pose[:3,3]=[dx*np.cos(np.radians(origin['lat'])),dy,xmp['RelativeAltitude']]
    f=a['focal']*np.hypot(W,H)/np.hypot(36,24)
    return Camera(str(a['frame']),np.array([[f,0,(W-1)/2],[0,f,(H-1)/2],[0,0,1]]),pose,W,H)


def freeze():
    if (OUT/'sites.json').exists():
        raise FileExistsError(OUT/'sites.json')
    heads={r['frame']:r for r in map(read,sorted((CACHE/'headers').glob('*.json')))}
    if len(heads)!=sum('-org-img/' in e[0] for e in read(CACHE/'archive.json')['entries']):
        raise ValueError('Incomplete camera index')
    old=read(DEV.parent/'all_labeled_cameras/plans.json')['plans']+read(DEV/'camera_choice/options.json')
    seen={i for p in old for i in (p['source'],*p['targets'])}
    seen.update(i for i in read(DEV/'shared_surface_renderer/protocol.json')['helpers'].values() if i)
    seen.update([7330,7331,7332])
    prior=[heads[i] for i in seen if i in heads and heads[i]['lat'] is not None]
    eligible={}
    for i,r in heads.items():
        if (not r['focal'] or r['lat'] is None or r['lon'] is None or len(r['xmp'])!=4
            or r['xmp']['RelativeAltitude']<=0 or r['xmp']['GimbalPitchDegree']>-68):
            continue
        with Image.open(labels_path(i)) as im:
            if im.size!=(r['width'],r['height']) or r['width']*3!=r['height']*4:
                continue
        if min(distance(r,b) for b in prior)>=500:
            eligible[i]=r
    slots=[]
    for i,a in eligible.items():
        if i%4 not in (0,1):
            continue
        near=[]
        for j in range(i-60,i+61):
            if j not in eligible or i==j:
                continue
            b=eligible[j]
            d=distance(a,b)
            yaw=abs((a['xmp']['GimbalYawDegree']-b['xmp']['GimbalYawDegree']+180)%360-180)
            if 3<=d<=45 and max(yaw,abs(a['xmp']['GimbalPitchDegree']-b['xmp']['GimbalPitchDegree']))<=20:
                near.append((d,j))
        refs=sorted(item for item in near if item[1]%4 in (0,1))
        targets=[j for _,j in sorted(near) if j%4 in (2,3)][:5]
        options=[list(pair) for pair in itertools.combinations(sorted(targets),2)
                 if 5<=distance(eligible[pair[0]],eligible[pair[1]])<=40]
        if refs and options:
            # Match the target separation distribution within each site.
            bins=[[pair for pair in options if low<=distance(*(eligible[t] for t in pair))<low+10]
                  for low in (5,15,25,35)]
            options=max(bins,key=len)
            slots.append((i,refs[0][1],options))
    sites=[]
    for i,aux,options in sorted(slots,key=lambda s:(-len(s[2]),s[0])):
        a=eligible[i]
        if any(distance(a,eligible[s['source']])<500 for s in sites):
            continue
        m=labels(i)
        if min((m==1).mean(),(m==2).mean())<.001:
            continue
        ids=sorted({i,aux,*[t for pair in options for t in pair]})
        sites.append(dict(source=i,helper=aux,options=options,
             min_dev_distance_m=min(distance(eligible[j],b) for j in ids for b in prior),
             cameras={j:camera_dict(camera(eligible[j],a)) for j in ids},
             source_mask_sha256={j:sha256(labels_path(j)) for j in (i,aux)},
             metadata={j:eligible[j] for j in ids}))
    write_json(OUT/'sites.json',dict(headers=len(heads),calibrated_remote=len(eligible),candidate_sources=len(slots),
        dev_frames=sorted(seen),sites=sites,code=sha256(Path(__file__)),archive=sha256(CACHE/'archive.json')))
    print([(s['source'],len(s['options'])) for s in sites])


def fetch(role):
    sites=read(OUT/'sites.json')['sites']
    rec={}
    for s in sites:
        ids=[s['source'],s['helper']] if role=='source' else sorted({i for pair in s['options'] for i in pair})
        rec.update({i:s['metadata'][str(i)] for i in ids})
    url=signed_url()
    def load(group):
        with RangeArchive(url) as ar,zipfile.ZipFile(ar) as z:
            for i in group:
                r=rec[i]
                p=CACHE/'images'/f'{i}.jpg'
                if p.exists() and zlib.crc32(p.read_bytes())==r['zip_crc32']:
                    continue
                blob=safe_member(z,r['member'])
                if zlib.crc32(blob)!=r['zip_crc32']:
                    raise ValueError(r['member'])
                tmp=p.with_suffix('.part')
                tmp.write_bytes(blob)
                tmp.replace(p)
    with ThreadPoolExecutor(max_workers=6) as pool:
        ids=sorted(rec)
        list(pool.map(load,[ids[k::6] for k in range(min(6,len(ids)))]))
    write_json(OUT/f'{role}_images.json',{i:sha256(CACHE/'images'/f'{i}.jpg') for i in rec})
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze','source','target'])
    stage=p.parse_args().stage
    freeze() if stage=='freeze' else fetch(stage)
