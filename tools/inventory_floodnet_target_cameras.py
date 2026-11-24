"""Survey target cameras independently of target scene classes using JPEG ZIP prefixes."""
import io
import itertools
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_floodnet_grass_strips import BASE, SITES, read, sha256, write_json
from audit_relation_floodnet_triplet import URL
from relation_nvs.archive import RangeArchive

OUT=BASE/'all_labeled_cameras'


def frame(row):
    return int(Path(row[0]).stem)


def header(block,entry):
    name,offset,size,raw,crc,method,flags=entry
    ns,extra=struct.unpack_from("<HH",block,26)
    if block[:4]!=b'PK\x03\x04' or method!=8 or block[30:30+ns].decode()!=name:
        raise ValueError("ZIP local header mismatch")
    data=zlib.decompressobj(-15).decompress(block[30+ns+extra:])
    image=Image.open(io.BytesIO(data))
    exif=image.getexif()
    gps=exif.get_ifd(34853)
    def coord(k):
        deg,minute,sec=map(float,gps[k])
        return (deg+minute/60+sec/3600)*(-1 if gps[k-1] in ("W","S") else 1)
    xmp={k.decode():float(v) for k,v in re.findall(rb'drone-dji:(RelativeAltitude|GimbalYawDegree|GimbalPitchDegree|GimbalRollDegree)="([+\-0-9.]+)"',data)}
    return dict(frame=frame(entry),width=image.width,height=image.height,lat=coord(2) if 2 in gps else None,
        lon=coord(4) if 4 in gps else None,xmp=xmp,focal=exif.get_ifd(34665).get(41989),
        member=name,zip_crc32=crc,metadata_prefix_bytes=len(block))


def worker(entries):
    missing=[e for e in entries if not (OUT/f"{frame(e)}.json").exists()]
    if not missing:
        return
    with RangeArchive(URL) as archive:
        for entry in missing:
            for length in (65536,262144,1048576):
                archive.seek(entry[1])
                try:
                    result=header(archive.read(min(entry[2]+256,length)),entry)
                    break
                except OSError:
                    if length==1048576:
                        raise
            write_json(OUT/f"{frame(entry)}.json",result)


def main():
    OUT.mkdir(exist_ok=True)
    entries=[e for e in read(BASE/'labeled_zip_inventory.json') if '/image/' in e[0]]
    inventory=read(SITES/'inventory.json')
    positives={r['frame']:r for r in inventory['records'] if r['calibrated']}
    mask_shapes={r['frame']:list(reversed(r['shape'])) for r in read(BASE/'mask_inventory/results.json')['records']}
    write_json(OUT/'protocol.json',dict(code=sha256(Path(__file__)),archive=URL,enumeration="all labeled JPEG headers, target contents not used to choose pairs",
        rules="source road/water positive and calibrated; target native grid, 3-35m, 20 degree orientation and 60 IDs; target separation >=3m",
        metadata_only="verify ZIP header name and geometry; full CRC separately before scoring any selected images"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker,[entries[i::8] for i in range(8)]))
    rows=[read(OUT/f'{frame(e)}.json') for e in entries]
    valid=[r for r in rows if r['lat'] is not None and r['lon'] is not None and len(r['xmp'])==4 and
           r['xmp']['RelativeAltitude']>0 and r['focal'] and [r['width'],r['height']]==mask_shapes[r['frame']]]
    def dist(a,b):
        return np.linalg.norm(np.radians([(a['lon']-b['lon'])*np.cos(np.radians(a['lat'])),a['lat']-b['lat']])*6378137)
    plans=[]
    for a in valid:
        if a['frame'] not in positives:
            continue
        targets=[]
        for b in valid:
            yaw=abs((a['xmp']['GimbalYawDegree']-b['xmp']['GimbalYawDegree']+180)%360-180)
            pitch=abs(a['xmp']['GimbalPitchDegree']-b['xmp']['GimbalPitchDegree'])
            if abs(a['frame']-b['frame'])<=60 and 3<=dist(a,b)<=35 and max(yaw,pitch)<=20:
                targets.append(b)
        for b,c in itertools.combinations(targets,2):
            if dist(b,c)>=3:
                plans.append(dict(source=a['frame'],targets=sorted([b['frame'],c['frame']])))
    write_json(OUT/'plans.json',dict(available_cameras=len(valid),total=len(entries),plans=plans))
    print(len(valid),len(plans),sorted(set(p['source'] for p in plans)))


if __name__=='__main__':
    main()
