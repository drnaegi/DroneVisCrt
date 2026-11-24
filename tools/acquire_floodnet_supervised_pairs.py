"""Acquire supervised FloodNet validation/test camera headers, source masks and selected complete views."""
import argparse
import concurrent.futures
import io
import itertools
from pathlib import Path
import struct
import zipfile
import zlib

import numpy as np
import requests
from PIL import Image

from inventory_floodnet_target_cameras import header
from evaluate_floodnet_grass_strips import BASE, read, sha256, write_json
from relation_nvs.archive import RangeArchive, safe_member

OUT=BASE/'supervised_flight_evaluation'
URL='https://www.kaggle.com/api/v1/datasets/download/i220854muhammadmomin/floodnet-supervised-v1-0'


def signed_url():
    with requests.get(URL,headers={'Range':'bytes=0-31'},stream=True,timeout=25) as r:
        r.raise_for_status()
        return r.url


def distance(a,b):
    return float(np.linalg.norm(np.radians([(a['lon']-b['lon'])*np.cos(np.radians(a['lat'])),a['lat']-b['lat']])*6378137))


def index():
    OUT.mkdir(exist_ok=True)
    url=signed_url()
    with RangeArchive(url) as ar, zipfile.ZipFile(ar) as archive:
        rows=[[z.filename,z.header_offset,z.compress_size,z.file_size,z.CRC,z.compress_type,z.flag_bits] for z in archive.infolist()
              if z.filename.startswith(('val/','test/'))]
        write_json(OUT/'members.json',dict(url=URL,archive_size=ar.size,members=rows))
        archive_size=ar.size
        for split in ('val','test'):
            masks=[e for e in rows if 'label-img' in e[0] and e[0].startswith(split+'/')]
            low=min(e[1] for e in masks)
            high=max(e[1]+e[2]+512 for e in masks)
            ar.seek(low)
            chunk=ar.read(high-low)
            maskdir=OUT/'masks'
            maskdir.mkdir(exist_ok=True)
            for e in masks:
                start=e[1]-low
                ns,extra=struct.unpack_from('<HH',chunk,start+26)
                name=chunk[start+30:start+30+ns].decode()
                start+=30+ns+extra
                if name!=e[0]:
                    raise ValueError(name)
                data=zlib.decompress(chunk[start:start+e[2]],-15)
                if zlib.crc32(data)!=e[4] or len(data)!=e[3]:
                    raise ValueError(name)
                (maskdir/Path(name).name).write_bytes(data)
    imgs=[e for e in rows if 'org-img' in e[0]]
    dest=OUT/'headers'
    dest.mkdir(exist_ok=True)
    def group(entries):
        with RangeArchive(url) as ar:
            if ar.size!=archive_size:
                raise ValueError("Changed archive length")
            for e in entries:
                p=dest/(Path(e[0]).stem+'.json')
                if p.exists():
                    continue
                for size in (65536,262144,1048576):
                    ar.seek(e[1])
                    try:
                        r=header(ar.read(min(size,e[2]+512)),e)
                        break
                    except OSError:
                        if size==1048576:
                            raise
                write_json(p,r)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(group,[imgs[i::12] for i in range(12)]))
    print(len(rows),len(imgs))


def plans():
    # Freeze regional method and reference-camera selection before scoring any target.
    old=read(BASE/'all_labeled_cameras/plans.json')['plans']
    prior_ids={i for p in old for i in [p['source'],*p['targets']]}
    prior=[read(BASE/'all_labeled_cameras'/f'{i}.json') for i in prior_ids]
    write_json(OUT/'protocol.json',dict(region_method=sha256(Path(__file__).with_name('evaluate_floodnet_region_pairs.py')),
        code=sha256(Path(__file__)),sites="minimum 500 m camera-centre distance from every previously evaluated FloodNet control",
        source_filter="positive road and water each >=0.1% source pixels at full native resolution",
        cameras="within same published split, source-to-target 3-35m, yaw/pitch diff <=20 deg, ID distance <=60; targets >=3m apart",
        budget="one pair per source: largest target-target separation; ties by frame ID; selection uses only target cameras and native grid dimensions",
        provenance="supervised dataset mirror; val/test fixed partitions; nearby images and source footprints may overlap across sources and evaluator training",
        mask_access="target size for all; class counts computed per prospective source; target class counts never enter its selection as a target"))
    headers=[read(p) for p in sorted((OUT/'headers').glob('*.json'))]
    eligible=[]
    for r in headers:
        with Image.open(OUT/'masks'/f"{r['frame']}_lab.png") as mask:
            if (r['width'],r['height'])==mask.size and min(len(r['xmp']),4)==4 and r['xmp']['RelativeAltitude']>0 and r['lat'] is not None and r['lon'] is not None and r['focal']:
                if min(distance(r,q) for q in prior)>=500:
                    eligible.append(r)
    pairs=[]
    for a in eligible:
        others=[]
        for b in eligible:
            if a['member'].split('/')[0]!=b['member'].split('/')[0] or abs(a['frame']-b['frame'])>60:
                continue
            yaw=abs((a['xmp']['GimbalYawDegree']-b['xmp']['GimbalYawDegree']+180)%360-180)
            pitch=abs(a['xmp']['GimbalPitchDegree']-b['xmp']['GimbalPitchDegree'])
            if max(yaw,pitch)<=20 and 3<=distance(a,b)<=35:
                others.append(b)
        options=sorted(((-distance(b,c),*sorted([b['frame'],c['frame']])) for b,c in itertools.combinations(others,2) if distance(b,c)>=3))
        if not options:
            continue
        with Image.open(OUT/'masks'/f"{a['frame']}_lab.png") as im:
            mask=np.asarray(im)
            if mask.ndim!=2 or np.max(mask)>9:
                raise ValueError("Unexpected labels")
            counts=np.bincount(mask.ravel(),minlength=10)
            if min(sum(counts[3:5]),counts[5])<.001*mask.size:
                continue
        gap,b,c=options[0]
        pairs.append(dict(source=a['frame'],targets=[b,c],target_distance_m=-gap,eligible_pairs=len(options),split=a['member'].split('/')[0]))
    write_json(OUT/'plans.json',dict(eligible_cameras=len(eligible),pairs=pairs))
    print("scheduled",len(pairs),flush=True)
    ids={i for p in pairs for i in [p['source'],*p['targets']]}
    entries=[r for r in eligible if r['frame'] in ids]
    folder=OUT/'images'
    folder.mkdir(exist_ok=True)
    url=signed_url()
    def fetch(group):
        with RangeArchive(url,max_bytes=1024**3) as ar, zipfile.ZipFile(ar) as archive:
            for r in group:
                p=folder/Path(r['member']).name
                if not p.exists() or zlib.crc32(p.read_bytes())!=r['zip_crc32']:
                    data=safe_member(archive,r['member'])
                    temp=p.with_suffix('.tmp')
                    temp.write_bytes(data)
                    temp.replace(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch,[entries[i::8] for i in range(8)]))
    write_json(OUT/'images.json',{r['frame']:sha256(folder/Path(r['member']).name) for r in entries})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('index','plans'))
    globals()[parser.parse_args().stage]()
