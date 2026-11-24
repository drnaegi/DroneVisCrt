"""Freeze pose-only triples, then recover each pair's rendering datum from source SIFT."""
import argparse
from itertools import combinations
from pathlib import Path
import struct

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from relation_nvs.aerial_mosaic import mutual_matches
from relation_nvs.geometry import Camera
from relation_nvs.io import camera_dict, load_camera, sha256, write_json
from tools.acquire_mill19_rubble import DATA, REVISION
from tools.benchmark_stereo_transport import OUT, ROOT, read

SELECTION=OUT/'mill19_rubble_selection.json'


def source_cameras():
    folder=DATA/'sparse/0'
    count,camera_id,model,w,h,fx,fy,cx,cy=struct.unpack('<QiiQQ4d',(folder/'cameras.bin').read_bytes())
    if count!=1 or model!=1:
        raise ValueError('Expected released PINHOLE camera')
    k=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]])
    cameras={}
    with (folder/'images.bin').open('rb') as stream:
        count,=struct.unpack('<Q',stream.read(8))
        for _ in range(count):
            _,qw,qx,qy,qz,tx,ty,tz,intr=struct.unpack('<i7di',stream.read(64))
            name=bytearray()
            while True:
                ch=stream.read(1)
                if ch==b'\0':
                    break
                if not ch:
                    raise EOFError('Incomplete COLMAP image name')
                name.extend(ch)
            name=name.decode()
            n_points,=struct.unpack('<Q',stream.read(8))
            stream.seek(24*n_points,1)
            if Path(name).name!=name or intr!=camera_id:
                raise ValueError('Unexpected image reference')
            pose=np.eye(4)
            pose[:3,:3]=Rotation.from_quat([qx,qy,qz,qw]).as_matrix().T
            pose[:3,3]=-pose[:3,:3]@[tx,ty,tz]
            cameras[name]=Camera(Path(name).stem,k,pose,w,h)
        if len(cameras)!=count or stream.tell()!=(folder/'images.bin').stat().st_size:
            raise ValueError('Unexpected pose file layout')
    return cameras


def select(group):
    cams=source_cameras()
    names=sorted(cams)
    repeat=group.endswith("_repeat")
    blocked={n for row in read(SELECTION)["selected"] for n in row["names"]} if repeat else set()
    targets=[n for n in names[4 if repeat else 0::8] if n not in blocked]
    held=set(targets)
    sources=[n for n in names if n not in held and n not in blocked]
    poses=np.stack([cams[n].camera_to_world for n in sources])
    centers=poses[:,:3,3]
    _,_,axes=np.linalg.svd(centers-centers.mean(0),full_matrices=False)
    up=axes[-1]
    if np.median(poses[:,:3,2]@up)>0:
        up=-up
    right=axes[0]
    basis=np.stack([right,np.cross(up,right),up])
    def down(c):
        corners=np.array([[x,y,1.] for x in (0,c.width-1) for y in (0,c.height-1)])
        return np.all((corners@np.linalg.inv(c.intrinsics).T@c.camera_to_world[:3,:3].T)@up < -1e-6)
    available=np.array([down(cams[n]) for n in sources])
    eligible=[]
    for target in targets:
        cam=cams[target]
        if not down(cam):
            continue
        axis,pos=cam.camera_to_world[:3,2],cam.camera_to_world[:3,3]
        d=np.linalg.norm(centers-pos,axis=1)
        near=[i for i in np.argsort(d)[:20] if available[i] and poses[i,:3,2]@axis>=np.cos(np.deg2rad(25))]
        pairs=[]
        for a,b in combinations(near,2):
            v=centers[b]-centers[a]
            fraction=(pos-centers[a])@v/max(v@v,1e-12)
            if .1<=fraction<=.9:
                off=np.linalg.norm(pos-centers[a]-fraction*v)
                pairs.append((off+.25*(d[a]+d[b]),sources[a],sources[b],float(fraction),off))
        if pairs:
            cost,a,b,fraction,off=min(pairs)
            eligible.append(dict(names=[a,b,target],cost=cost,fraction=fraction,offset_units=off))
    indices=np.rint(np.linspace(0,len(eligible)-1,min(16,len(eligible)))).astype(int)
    if not len(indices):
        raise ValueError('No eligible target')
    write_json(OUT/(group+"_selection.json"),dict(revision=REVISION,registered=len(cams),candidate_targets=targets,
               eligible=[r['names'][2] for r in eligible],selected=[eligible[i] for i in indices],
               source_world_basis=basis.tolist(),protocol_sha256=sha256(OUT/('protocol_pointmap_source_fit.md' if repeat else 'protocol_mill19_rubble.md')),
               metadata_sha256={f:sha256(DATA/'sparse/0'/f) for f in ('images.bin','cameras.bin')}))
    print(len(eligible),len(indices))


def canonical_image(name,cam):
    width,height=512,384
    k=cam.intrinsics
    f=1.01*max(k[0,0]*width/cam.width,k[1,1]*height/cam.height)
    nk=np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
    d=DATA/'canonical'/Path(name).with_suffix('.png')
    if not d.exists():
        d.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(DATA/'images'/name) as im:
            if im.size!=(cam.width,cam.height):
                raise ValueError('Original image differs from calibration')
            # Resize a continuous crop. Pillow BOX positions use pixel-edge coordinates.
            scale=k.diagonal()[:2]/f
            span=np.array([width,height])*scale
            center=k[:2,2]+.5
            box=tuple([*(center-span/2),*(center+span/2)])
            im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS,box=box).save(d)
    return d,Camera(cam.camera_id,nk,cam.camera_to_world,width,height)


def datum(cams,images,basis):
    xy=mutual_matches(*images)
    if len(xy[0])<8:
        raise ValueError('Too few two-source matches')
    ps=[c.intrinsics@np.linalg.inv(c.camera_to_world)[:3] for c in cams[:2]]
    q=cv2.triangulatePoints(*ps,xy[0].T,xy[1].T)
    good=np.abs(q[3])>1e-10
    q=q/np.where(good,q[3],1)
    for p,points in zip(ps,xy):
        h=p@q
        good&=(h[2]>0)&(np.linalg.norm(h[:2]/np.where(h[2]>0,h[2],1)-points.T,axis=0)<=2)
    if good.sum()<8:
        raise ValueError('Too few consistent triangulations')
    z=float(np.median(basis[2]@q[:3,good]))
    centers=np.stack([basis@c.camera_to_world[:3,3] for c in cams])
    heights=centers[:,2]-z
    if heights.min()<=0:
        raise ValueError('Reference plane above camera')
    factor=50/heights[:2].mean()
    origin=np.r_[centers[:2,:2].mean(0),z]
    def convert(c):
        pose=c.camera_to_world.copy()
        pose[:3,:3]=basis@pose[:3,:3]
        pose[:3,3]=factor*(basis@pose[:3,3]-origin)
        return Camera(c.camera_id,c.intrinsics,pose,c.width,c.height)
    return [camera_dict(convert(c)) for c in cams],dict(matches_used=int(good.sum()),scale=factor,
               reference_plane=z,mean_source_height=50.,known_absolute_scale=False)


def prepare(group):
    sel=read(OUT/(group+"_selection.json"))
    original=source_cameras()
    rows,failures=[],{}
    for entry in sel['selected']:
        names=entry['names']
        sources=[canonical_image(n,original[n]) for n in names[:2]]
        # The target RGB has no role in the SIFT fit or the camera transformation.
        target=original[names[2]]
        tc=Camera(target.camera_id,sources[0][1].intrinsics,target.camera_to_world,512,384)
        try:
            images=[np.asarray(Image.open(p)) for p,_ in sources]
            cameras,fit=datum([*[c for _,c in sources],tc],images,np.array(sel['source_world_basis']))
        except ValueError as e:
            failures[names[2]]=str(e)
            continue
        truth,_=canonical_image(names[2],target)
        paths=[p for p,_ in sources]+[truth]
        rows.append(dict(paths=[str(p.relative_to(ROOT)) for p in paths],cameras=cameras,datum=fit,
                         original_names=names,original_sha256=[sha256(DATA/'images'/n) for n in names]))
    write_json(OUT/(group+'_cases.json'),rows)
    write_json(OUT/(group+'_eligibility.json'),dict(selected=len(sel['selected']),completed=len(rows),failed=failures))
    print(len(rows),failures)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['select','prepare'])
    cv2.setNumThreads(2)
    p.add_argument('--repeat',action='store_true')
    args=p.parse_args()
    globals()[args.stage]('mill19_rubble_repeat' if args.repeat else 'mill19_rubble')
