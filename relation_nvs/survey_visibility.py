"""Visible-surface rasterization for the survey renderer, retaining original source UVs."""
import cv2
import numpy as np
import torch
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes

from .aerial_mosaic import plane_projection, transform
from .stereo_transport import project, sample


def raster_source(xy, depth, image, target):
    """Rasterize a depth-discontinuity-filtered triangular depth image onto the target."""
    height, width = depth.shape
    ys = np.unique(np.r_[np.arange(0, height, 2),height-1])
    xs = np.unique(np.r_[np.arange(0, width, 2),width-1])
    v, u = np.meshgrid(ys,xs,indexing='ij')
    xy, z = xy[v,u], depth[v,u]
    scale = min(target.height, target.width)/2.
    verts = np.stack(((target.width/2-.5-xy[...,0])/scale,
                      (target.height/2-.5-xy[...,1])/scale,z),axis=-1).reshape(-1,3)
    indices = np.arange(z.size).reshape(z.shape)
    a,b,c,d = indices[:-1,:-1],indices[:-1,1:],indices[1:,:-1],indices[1:,1:]
    tris = np.concatenate((np.stack([a,b,d],axis=-1).reshape(-1,3),
                           np.stack([a,d,c],axis=-1).reshape(-1,3)))
    corner_z = verts[tris,2]
    keep = (corner_z.min(axis=1)>0) & (corner_z.max(axis=1)<1.10*corner_z.min(axis=1))
    h,w = target.height,target.width
    if not keep.any():
        return np.zeros((h,w,3),np.uint8),np.full((h,w),np.inf),np.full((h,w),-np.inf)
    faces = torch.as_tensor(tris[keep],device='cuda',dtype=torch.long)
    mesh = Meshes([torch.as_tensor(verts,device='cuda',dtype=torch.float32)],[faces])
    ids, _, bary, _ = rasterize_meshes(mesh,image_size=(h,w),faces_per_pixel=1,
                                      perspective_correct=True,cull_backfaces=False,bin_size=32,
                                      max_faces_per_bin=50000)
    index = ids[0,...,0]
    face = faces[index.clamp_min(0)].cpu().numpy()
    weights = bary[0,...,0,:].cpu().numpy()
    source_uv = np.stack((u,v),axis=-1).reshape(-1,2)
    uv = (source_uv[face]*weights[...,None]).sum(axis=-2).astype(np.float32)
    predicted_z = (verts[face,2]*weights).sum(axis=-1)
    hit = index.cpu().numpy()>=0
    margin = np.minimum.reduce((uv[...,0],uv[...,1],width-1-uv[...,0],height-1-uv[...,1]))
    return sample(image,uv),np.where(hit,predicted_z,np.inf),np.where(hit,margin,-np.inf)


@torch.inference_mode()
def render_surfaces(atlas,target):
    fallback, provenance, _ = atlas.plane.render(target)
    colors, depths, margins = [], [], []
    distances = []
    for j,(camera,(uv,xyz,residual,trust,match)) in enumerate(zip(atlas.cameras,atlas.geometry)):
        other = atlas.cameras[1-j]
        base = other.camera_to_world[:3,3]-camera.camera_to_world[:3,3]
        delta = target.camera_to_world[:3,3]-camera.camera_to_world[:3,3]
        distances.append(np.linalg.norm(delta))
        fraction = np.clip(np.dot(delta,base)/max(np.dot(base,base),1e-9),-1.,2.)
        homography = plane_projection(target) @ np.linalg.inv(plane_projection(other))
        xy = project(xyz,target) + fraction*(transform(match,homography)-transform(match-residual,homography))
        depth = (xyz-target.camera_to_world[:3,3]) @ target.camera_to_world[:3,2]
        color, z, margin = raster_source(xy,depth,atlas.images[j],target)
        colors.append(color); depths.append(z); margins.append(margin)
    z = np.stack(depths)
    visible = np.isfinite(z) & (z <= 1.05*np.min(z,axis=0))
    dist = np.asarray(distances)/max(sum(distances),1e-9)
    quality = np.minimum(np.stack(margins),16)/16 - dist[:,None,None]
    best = np.argmax(np.where(visible,quality,-np.inf),axis=0)
    hit = visible.any(axis=0)
    hard = np.where(hit[...,None],np.where((best==0)[...,None],colors[0],colors[1]),fallback)
    weight = np.where(visible,np.exp(2*quality),0.)
    mix = (weight[...,None]*np.stack(colors)).sum(axis=0)/np.maximum(weight.sum(axis=0)[...,None],1e-12)
    blended = np.where(hit[...,None],mix.round().clip(0,255).astype(np.uint8),fallback)
    return hard,blended,np.where(hit,best+1,provenance).astype(np.uint8)


def combine_surface_and_inverse(surface, inverse):
    """Transfer source detail when both target predictions place edges consistently."""
    s,i = surface.astype(np.float32),inverse.astype(np.float32)
    # Estimate the unresolved displacement from local first-order brightness constancy.
    # Pool over a patch to avoid division by the derivative at a single pixel.
    xs = cv2.Sobel(i,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    ys = cv2.Sobel(i,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    def smooth(x):
        return cv2.GaussianBlur(x,(0,0),1.5)
    delta = i-s
    # Remove low-frequency color/exposure disagreement before the motion estimate.
    delta -= cv2.GaussianBlur(delta,(0,0),3.)
    error = smooth((delta**2).mean(axis=2))
    gradient = smooth((xs**2+ys**2).mean(axis=2))
    variance = error/(gradient+4.)
    aligned = np.exp(-.5*np.minimum(variance,16.))[...,None]
    # A common low-frequency rendering supplies coverage. Keep source high frequencies
    # only where the local projected position is compatible with that rendering.
    detail = i - cv2.GaussianBlur(i,(0,0),1.5)
    low_surface = cv2.GaussianBlur(s,(0,0),1.5)
    fixed = np.clip(low_surface+detail,0,255).round().astype(np.uint8)
    adaptive = low_surface + aligned*detail + (1-aligned)*(s-low_surface)
    return fixed,np.clip(adaptive,0,255).round().astype(np.uint8)
