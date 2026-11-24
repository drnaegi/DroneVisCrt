"""Use source RGB/labels and DJI telemetry to project a held-out pair of real UAV images."""
import argparse
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from probe_relation_local_transects import EVIDENCE, project, read, sha256, write_json, world_points
from probe_relation_spatial_neighbors import neighbors, surface_samples
from relation_nvs.geometry import Camera

BASE=EVIDENCE/"independent_data_intake"
DATA=BASE/"positive_triplet"
OUT=DATA/"transfer"
W,H=512,384


def calibration(path):
    image=Image.open(path)
    exif=image.getexif()
    gps=exif.get_ifd(34853)
    def coordinate(dms,ref):
        deg,minute,sec=map(float,dms)
        return (deg+minute/60+sec/3600)*(-1 if ref in ("S","W") else 1)
    lat=coordinate(gps[2],gps[1]);lon=coordinate(gps[4],gps[3])
    xmp=dict((k.decode(),float(v)) for k,v in re.findall(
        rb'drone-dji:(RelativeAltitude|AbsoluteAltitude|GimbalYawDegree|GimbalPitchDegree|GimbalRollDegree)="([+\-0-9.]+)"',path.read_bytes()))
    yaw,pitch,roll=np.radians([xmp[k] for k in ("GimbalYawDegree","GimbalPitchDegree","GimbalRollDegree")])
    east=np.array([np.cos(yaw),-np.sin(yaw),0.])
    forward=np.array([np.sin(yaw)*np.cos(pitch),np.cos(yaw)*np.cos(pitch),np.sin(pitch)])
    down=np.cross(forward,east)
    rotation=np.column_stack((east*np.cos(roll)+down*np.sin(roll),down*np.cos(roll)-east*np.sin(roll),forward))
    focal=float(exif.get_ifd(34665)[41989])*np.hypot(W,H)/np.hypot(36.,24.)
    return lat,lon,xmp,rotation,focal


def cameras():
    ids=read(DATA/"selection.json")["triples"][0]
    params={i:calibration(DATA/f"{i}.jpg") for i in ids}
    origin=params[ids[1]]
    result={}
    for i,(lat,lon,xmp,rot,f) in params.items():
        c2w=np.eye(4)
        c2w[:3,:3]=rot
        c2w[:3,3]=[np.radians(lon-origin[1])*6378137*np.cos(np.radians(origin[0])),
                    np.radians(lat-origin[0])*6378137,xmp["RelativeAltitude"]]
        k=np.array([[f,0,(W-1)/2],[0,f,(H-1)/2],[0,0,1]])
        result[i]=Camera(f"frame{i}",k,c2w,W,H)
    return result


def plane(camera):
    y,x=np.indices((H,W))
    xy=np.column_stack((x.ravel(),y.ravel()))
    rays=np.column_stack((xy,np.ones(len(xy))))@np.linalg.inv(camera.intrinsics).T@camera.camera_to_world[:3,:3].T
    z=-camera.camera_to_world[2,3]/rays[:,2]
    z[z<=0]=np.nan
    return xy,z


def data(frame):
    image=np.asarray(Image.open(DATA/f"{frame}.jpg").resize((W,H),Image.Resampling.LANCZOS))
    raw=np.asarray(Image.open(BASE/"mask_inventory"/f"{frame}_lab.png").resize((W,H),Image.Resampling.NEAREST))
    labels=np.zeros_like(raw);labels[np.isin(raw,[3,4])]=1;labels[raw==5]=2
    return image,labels


def generate():
    OUT.mkdir(exist_ok=False)
    cams=cameras()
    ids=sorted(cams)
    source=ids[1]
    write_json(OUT/"protocol.json",dict(source=source,targets=[ids[0],ids[-1]],
        code_sha256=sha256(Path(__file__)),source_modalities="real center RGB, annotated road/water and EXIF/XMP",
        camera_assumptions="ENU tangent plane, relative height above horizontal take-off plane, principal point at center, equivalent focal length diagonal",
        no_measured_depth=True,goal="frozen 15 m source relation eligibility plus external RGB/mask geometry audit",
        scoring="full and common-support RGB MAE, road and water mask IoU, observed source-to-target correspondence residuals",
        split="external real data qualification; source/target mask co-presence used for finding the clip"))
    img,labels=data(source)
    xy,z=plane(cams[source])
    anchors=neighbors(labels,z.reshape(H,W),cams[source])
    _,road=surface_samples(labels,z.reshape(H,W),cams[source],1)
    _,water=surface_samples(labels,z.reshape(H,W),cams[source],2)
    minimum=float(cKDTree(road).query(water)[0].min())
    result={}
    for i in (ids[0],ids[-1]):
        uv,d=plane(cams[i])
        world=world_points(cams[i],uv,d)
        mapping,source_z=project(cams[source],world)
        mx,my=mapping.T.astype(np.float32).reshape(2,H,W)
        ok=((mx>=0)&(mx<=W-1)&(my>=0)&(my<=H-1)&(source_z.reshape(H,W)>0)).astype(np.uint8)
        rgb=cv2.remap(img,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        seg=cv2.remap(labels,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
        dest=OUT/str(i);dest.mkdir()
        files={}
        for name,array in (("warped",rgb),("source_copy",img),("labels",seg),("support",ok)):
            p=dest/f"{name}.png";Image.fromarray(array).save(p);files[name]=sha256(p)
        result[str(i)]=dict(hashes=files,translation_m=float(np.linalg.norm(cams[i].camera_to_world[:3,3]-cams[source].camera_to_world[:3,3])))
    write_json(OUT/"generated.json",dict(anchors=anchors,minimum_surface_distance_m=minimum,targets=result))


def audit():
    cams=cameras()
    source=sorted(cams)[1]
    original,classes=data(source)
    state=read(OUT/"generated.json")
    sift=cv2.SIFT_create(nfeatures=3500)
    rows=[]
    for key,record in state["targets"].items():
        dest=OUT/key
        if any(sha256(dest/f"{k}.png")!=v for k,v in record["hashes"].items()):
            raise ValueError("Changed transfer")
        truth,labels=data(int(key))
        support=np.asarray(Image.open(dest/"support.png")).astype(bool)
        def features(image):
            pts,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
            return np.array([kp.pt for kp in pts]),desc
        p,da=features(original);q,db=features(truth)
        matcher=cv2.BFMatcher()
        def ratio(a,b):
            return {(m.queryIdx,m.trainIdx) for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.75*n.distance}
        ties=sorted(ratio(da,db)&{(j,i) for i,j in ratio(db,da)})
        p=p[[i for i,j in ties]];q=q[[j for i,j in ties]]
        direction=np.column_stack((p,np.ones(len(p))))@np.linalg.inv(cams[source].intrinsics).T@cams[source].camera_to_world[:3,:3].T
        z=-cams[source].camera_to_world[2,3]/direction[:,2]
        projected,_=project(cams[int(key)],world_points(cams[source],p,z))
        def residual(xy):
            return np.percentile(np.linalg.norm(xy-q,axis=1),[50,90]).tolist()
        metrics={}
        for name,pred,mask in (("source_copy",original,classes),
                                ("warped",np.asarray(Image.open(dest/"warped.png")),np.asarray(Image.open(dest/"labels.png")))):
            err=np.abs(pred.astype(float)-truth)/255
            metrics[name]=dict(full_mae=float(err.mean()),supported_mae=float(err[support].mean()),
                full_iou={str(c):float(((mask==c)&(labels==c)).sum()/((mask==c)|(labels==c)).sum()) for c in (1,2)})
        rows.append(dict(target=key,translation_m=record["translation_m"],coverage=float(support.mean()),
            mutual_ties=len(ties),copy_median_p90_px=residual(p),telemetry_median_p90_px=residual(projected),metrics=metrics))
    write_json(OUT/"results.json",dict(source_anchors=len(state["anchors"]),minimum_surface_distance_m=state["minimum_surface_distance_m"],
        rows=rows,semantics="warped source mask compared with target annotation; generated RGB not independently relabeled"))
    print(read(OUT/"results.json"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("generate","audit"))
    (generate if parser.parse_args().stage=="generate" else audit)()
