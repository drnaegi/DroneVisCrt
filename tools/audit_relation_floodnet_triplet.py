"""Qualify real UAV triplets for road/water evaluation using GPS availability and held-out ties."""
import io
import itertools
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from probe_relation_local_transects import EVIDENCE, read, sha256, write_json
from relation_nvs.archive import RangeArchive, safe_member

OUT=EVIDENCE/"independent_data_intake"/"floodnet_triplet"
URL="https://hf-mirror.com/datasets/torchgeo/floodnet/resolve/c5ea256ffb7832d5ae78cda3f4b812521faf32c5/floodnet.zip"


def main():
    OUT.mkdir(exist_ok=True)
    indices=read(OUT.parent/"floodnet_members.json")
    prefix="floodnet/Train/Labeled/Flooded"
    ids=sorted(int(Path(n).stem) for n in indices if n.startswith(prefix+"/image/") and n.endswith(".jpg"))
    runs=[list(v for _,v in group) for _,group in itertools.groupby(enumerate(ids),lambda r:r[1]-r[0])]
    selected=sorted(runs,key=lambda r:(-len(r),r[0]))[0][:3]
    write_json(OUT/"protocol.json",dict(url=URL,archive_member_index_sha256=sha256(OUT.parent/"floodnet_members.json"),
        selection="first three frames from longest contiguous labeled Flooded training run, tie by smallest ID",
        selected=selected,scope="cross-dataset qualification; camera geometry and content overlap unverified",
        match_rule="960px long edge, mutual SIFT ratio 0.75, spatial checkerboard 64-pixel cells, fit homography on one half",
        qualification="at least 12 validation matches within 3 px in >=4 spatial cells for each adjacent pair",
        code_sha256=sha256(Path(__file__))))
    raw=OUT/"files"
    raw.mkdir(exist_ok=True)
    members=[]
    with RangeArchive(URL,max_bytes=100*1024**2) as remote,zipfile.ZipFile(remote) as archive:
        for n in selected:
            for kind,filename in (("image",f"{n}.jpg"),("mask",f"{n}_lab.png")):
                name=f"{prefix}/{kind}/{filename}"
                dest=raw/filename
                if not dest.exists():
                    data=safe_member(archive,name)
                    dest.write_bytes(data)
                import zlib
                if zlib.crc32(dest.read_bytes())!=archive.getinfo(name).CRC:
                    raise ValueError("Unexpected member contents")
                members.append(dict(member=name,sha256=sha256(dest),crc32=archive.getinfo(name).CRC,size=dest.stat().st_size))
    sift=cv2.SIFT_create(nfeatures=3500)
    images={}
    facts={}
    for n in selected:
        image=Image.open(raw/f"{n}.jpg").convert("RGB")
        mask=np.asarray(Image.open(raw/f"{n}_lab.png"))
        exif=Image.open(raw/f"{n}.jpg").getexif()
        reduced=np.asarray(image.resize((960,round(image.height*960/image.width)),Image.Resampling.LANCZOS))
        points,desc=sift.detectAndCompute(cv2.cvtColor(reduced,cv2.COLOR_RGB2GRAY),None)
        images[n]=(np.array([p.pt for p in points]),desc,reduced)
        facts[n]=dict(size=image.size,mask_shape=mask.shape,exif_gps_present=bool(exif.get_ifd(34853)),
                      original_focal_length_available=37386 in exif.get_ifd(34665),
                      class_pixels={str(int(a)):int(b) for a,b in zip(*np.unique(mask,return_counts=True))})
    pairs=[]
    cv2.setRNGSeed(123)
    matcher=cv2.BFMatcher()
    for a,b in itertools.combinations(selected,2):
        first,da,_=images[a]
        second,db,_=images[b]
        def ratio(da,db):
            return {(m.queryIdx,m.trainIdx) for m,n in matcher.knnMatch(da,db,k=2) if m.distance<.75*n.distance}
        matches=sorted(ratio(da,db)&{(j,i) for i,j in ratio(db,da)})
        p=first[[i for i,j in matches]]
        q=second[[j for i,j in matches]]
        cells=(p//64).astype(int)
        train=cells.sum(axis=1)%2==0
        homography=None
        inliers=0
        valid_cells=0
        if sum(train)>=8 and sum(~train)>=1:
            homography,_=cv2.findHomography(p[train],q[train],cv2.RANSAC,3.)
            if homography is not None:
                prediction=cv2.perspectiveTransform(p[~train,None,:],homography)[:,0,:]
                close=np.linalg.norm(prediction-q[~train],axis=1)<=3.
                inliers=int(close.sum())
                valid_cells=len(np.unique(cells[~train][close],axis=0))
        pairs.append(dict(first=a,second=b,mutual_matches=len(matches),validation_within_3px=inliers,
                          supported_cells=valid_cells,qualified=inliers>=12 and valid_cells>=4))
    panel=Image.new("RGB",(len(selected)*320,240))
    for i,n in enumerate(selected):
        img=Image.fromarray(images[n][2]);img.thumbnail((320,240));panel.paste(img,(i*320,0))
    panel.save(OUT/"contact.png")
    write_json(OUT/"results.json",dict(members=members,images=facts,pairs=pairs,split="published Train/Labeled/Flooded",
        adjacent_pairs_qualified=all(r["qualified"] for r in pairs if r["second"]==r["first"]+1)))
    print(facts,pairs)


if __name__=="__main__":
    main()
