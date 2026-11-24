"""Estimate convex frequency-band mixing weights from frozen camera and image features."""
import cv2
import numpy as np

from .aerial_mosaic import plane_projection, transform
from .stereo_transport import grid_for


def decompose(image):
    fine = cv2.GaussianBlur(image,(0,0),1.5)
    coarse = cv2.GaussianBlur(image,(0,0),4.)
    return coarse,fine-coarse,image-fine


def camera_features(cameras):
    target = cameras[-1]
    world = transform(grid_for(target),np.linalg.inv(plane_projection(target)))
    margins = []
    for source in cameras[:2]:
        uv = transform(world,plane_projection(source))
        margin = np.minimum.reduce((uv[...,0],uv[...,1],source.width-1-uv[...,0],source.height-1-uv[...,1]))
        margins.append(np.clip(margin/16.,-1.,1.))
    lo,hi = np.minimum(*margins), np.maximum(*margins)
    centers = np.stack([c.camera_to_world[:3,3] for c in cameras])
    axis = centers[1]-centers[0]
    delta = centers[2]-centers[0]
    frac = delta@axis/max(axis@axis,1e-12)
    offset = np.linalg.norm(delta-frac*axis)/max(centers[2,2],1e-9)
    return [lo,hi,np.full_like(lo,min(1.,offset)),np.full_like(lo,min(1.,abs(2*frac-1)))]


def band_inputs(surface, inverse, cameras):
    s,i = [im.astype(np.float32)/255. for im in (surface,inverse)]
    geometry = camera_features(cameras)
    for sb,ib in zip(decompose(s),decompose(i)):
        diff = ib-sb
        def energy(field):
            power = cv2.GaussianBlur(np.mean(field**2,axis=-1),(0,0),2.)
            return np.sqrt(np.maximum(0.,power))
        # At fixed pixel energy, branch contrast and disagreement predict which
        # estimate can retain a feature without shifting its position.
        es,ei = energy(sb),energy(ib)
        disagreement = energy(diff)/(es+ei+1./255)
        contrast = (es-ei)/(es+ei+1./255)
        photo = [np.ones_like(contrast),disagreement,contrast,disagreement*contrast]
        features = np.stack([*photo,*geometry,disagreement*geometry[0],contrast*geometry[0]],axis=-1)
        yield sb,diff,features.astype(np.float32)


FEATURE_COUNTS = {'constant':1, 'image':4, 'geometry':10}


def combine(surface,inverse,cameras,weights):
    result = np.zeros_like(surface,dtype=np.float32)
    for (base,delta,f),coeff in zip(band_inputs(surface,inverse,cameras),weights,strict=True):
        beta = np.asarray(coeff,dtype=np.float32)
        weight = np.clip(f[...,:len(beta)] @ beta,0,1)
        result += base+weight[...,None]*delta
    return np.clip(result*255,0,255).round().astype(np.uint8)
