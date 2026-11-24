"""Fit a common metric coordinate system to two frozen pointmaps using calibrated image rays."""
from contextlib import contextmanager
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from .stereo_transport import grid_for, sample

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_SHA = '7c300a89534113436bde52732d3151212bcbd90f0aa3c8d1496f86d84bfe4b42'


def default_checkpoint():
    local = ROOT/'checkpoints/DUSt3R/model.safetensors'
    return local if local.is_file() else Path('/autodl-tmp/DroneVisCrt/checkpoints/DUSt3R/model.safetensors')


@contextmanager
def pointmap_precision():
    # CroCo was released with TF32 matmuls. Scope its import-time global setting.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pointmap_precision()
def load_pair_model(device='cuda', checkpoint=None):
    sys.path.insert(0,str(ROOT/'code/viewcrafter-sparse/extern/dust3r'))
    from dust3r.model import AsymmetricCroCo3DStereo
    from safetensors.torch import load_file

    model = AsymmetricCroCo3DStereo(
        pos_embed='RoPE100', img_size=(512,512), head_type='dpt', output_mode='pts3d',
        depth_mode=('exp',-float('inf'),float('inf')), conf_mode=('exp',1,float('inf')),
        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768,
        dec_depth=12, dec_num_heads=12, patch_embed_cls='PatchEmbedDust3R',
        landscape_only=False, freeze='none')
    state = load_file(str(checkpoint or default_checkpoint()))
    for head in ('downstream_head1','downstream_head2'):
        for index in range(4):
            state[f'{head}.dpt.scratch.layer_rn.{index}.weight'] = state[f'{head}.dpt.scratch.layer{index+1}_rn.weight']
    model.load_state_dict(state,strict=True)
    return model.eval().to(device)


@torch.inference_mode()
@pointmap_precision()
def ordered_pointmaps(model, images):
    """Each directed pair shares a predicted coordinate system, with the first view at its origin."""
    h,w = images[0].shape[:2]
    if images[1].shape != images[0].shape:
        raise ValueError('Both source rasters must have the same dimensions')
    size = tuple(max(16,16*round(512*n/max(h,w)/16)) for n in (w,h))
    data = np.stack([cv2.resize(im,size,interpolation=cv2.INTER_AREA) for im in images])
    tensor = torch.from_numpy(data).permute(0,3,1,2).float().to(next(model.parameters()).device)/127.5-1.
    shape = torch.tensor([[size[1],size[0]]]*2,device=tensor.device)
    self_pred, peer_pred = model(dict(img=tensor,instance=['a','b'],true_shape=shape),
                                 dict(img=tensor.flip(0),instance=['b','a'],true_shape=shape))
    def at_size(data):
        return [cv2.resize(v,(w,h),interpolation=cv2.INTER_LINEAR) for v in data.cpu().float().numpy()]
    own,peer = at_size(self_pred['pts3d']),at_size(peer_pred['pts3d_in_other_view'])
    c1,c2 = at_size(self_pred['conf']),at_size(peer_pred['conf'])
    return [((own[0],peer[0]),(c1[0],c2[0])),((peer[1],own[1]),(c2[1],c1[1]))]


def ray_directions(uv,camera,reference):
    homogeneous = np.concatenate((uv,np.ones_like(uv[...,:1])),axis=-1)
    rays = homogeneous @ np.linalg.inv(camera.intrinsics).T @ camera.camera_to_world[:3,:3].T @ reference.camera_to_world[:3,:3]
    return rays / np.linalg.norm(rays,axis=-1,keepdims=True)


def _fit_similarity(h, proj, pc, weights, nominal, ridge):
    from scipy.spatial.transform import Rotation

    # Seven tangent directions for translation, rotation and uniform log scale.
    matrix = nominal.copy()
    skew = np.stack([np.cross(np.eye(3)[j],np.eye(3)).T for j in range(3)])
    for _ in range(5):
        residual = np.einsum('nij,nj->ni',proj,h@matrix.T)-pc
        err = np.linalg.norm(residual,axis=1)
        huber = max(.005,1.4826*np.median(err))
        w = weights*np.minimum(1.,huber/np.maximum(err,1e-9))
        basis = np.zeros((7,3,4))
        basis[:3,:,3] = np.eye(3)
        basis[3:6,:,:3] = skew@matrix[:,:3]
        basis[6,:,:3] = matrix[:,:3]
        db = basis.reshape(7,12).T
        projected = np.einsum('nij,ajk,nk->nia',proj,basis,h,optimize=True)
        flat = projected.reshape(-1,7)
        weighted = (w[:,None,None]*projected).reshape(-1,7)
        penalty = db.T*ridge
        d = np.linalg.lstsq(weighted.T@flat+penalty@db,
                            -weighted.T@residual.ravel()-penalty@(matrix-nominal).ravel(),rcond=None)[0]
        matrix[:,3] += d[:3]
        matrix[:,:3] = np.exp(np.clip(d[6],-1.,1.))*Rotation.from_rotvec(d[3:6]).as_matrix()@matrix[:,:3]
    return matrix


def ray_alignment(points, rays, centers, weights, affine, *, similarity=False):
    """Fit X=Aq+b by minimizing robust orthogonal distances to observed rays in baseline units."""
    q,r,c,w = [np.asarray(v,np.float64) for v in (points,rays,centers,weights)]
    valid = np.isfinite(q).all(axis=1) & np.isfinite(r).all(axis=1) & np.isfinite(c).all(axis=1) & np.isfinite(w) & (w>0)
    q,r,c,w = q[valid],r[valid],c[valid],w[valid]
    if len(q)<16:
        return None
    length = np.max(np.linalg.norm(c,axis=1))
    if length < 1e-8:
        return None
    c = c/length
    proj = np.eye(3)[None] - r[:,:,None]*r[:,None,:]
    pc = np.einsum('nij,nj->ni',proj,c)
    pq = np.einsum('nij,nj->ni',proj,q)
    # Scale with fixed reference origin is the nested, one-parameter camera model.
    scale = float(np.sum(w[:,None]*pq*pc)/max(np.sum(w[:,None]*pq**2),1e-12))
    if not np.isfinite(scale) or scale<=0:
        return None
    nominal = np.column_stack((np.eye(3)*scale,np.zeros(3)))
    if not affine and not similarity:
        return nominal*length
    h = np.column_stack((q,np.ones(len(q))))
    design = (proj[:,:,:,None]*h[:,None,None,:]).reshape(-1,3,12)
    beta = nominal.ravel().copy()
    # Weak prior toward uniform scale limits unsupported shears of nearly planar pointmaps.
    normalizer = np.mean(w[:,None]*h*h,axis=0)
    ridge = .001*len(w)*np.tile(normalizer,3)
    if similarity:
        return _fit_similarity(h,proj,pc,w,nominal,ridge)*length
    for _ in range(5):
        err = np.linalg.norm(np.einsum('nij,j->ni',design,beta)-pc,axis=1)
        huber = max(.005,1.4826*np.median(err))
        robust = w*np.minimum(1.,huber/np.maximum(err,1e-9))
        flat = design.reshape(-1,12)
        weighted = (design*robust[:,None,None]).reshape(-1,12)
        beta = np.linalg.lstsq(weighted.T@flat+np.diag(ridge),
                               weighted.T@pc.ravel()+ridge*nominal.ravel(),rcond=None)[0]
    return beta.reshape(3,4)*length


def calibrated_pair_depth(pair_maps,cameras,flows,rule):
    """Compare origin-and-scale, full ray fit, and a fit also anchored by reciprocal matches."""
    if rule not in ('scale','rays','matches','similarity'):
        raise ValueError(rule)
    grids = [grid_for(cam) for cam in cameras]
    predictions = [[],[]]
    diagnostics = []
    for ref,(maps,confidence) in enumerate(pair_maps):
        reference = cameras[ref]
        rotation = reference.camera_to_world[:3,:3]
        positions = [(c.camera_to_world[:3,3]-reference.camera_to_world[:3,3])@rotation for c in cameras]
        blocks = []
        for j,q in enumerate(maps):
            take = np.s_[::8,::8]
            coords = grids[j][take]
            weights = 1.-1./np.maximum(confidence[j][take],1.)
            def block(k,uv,weight):
                rays = ray_directions(uv,cameras[k],reference)
                blocks.append((q[take].reshape(-1,3),rays.reshape(-1,3),
                               np.broadcast_to(positions[k],rays.shape).reshape(-1,3),weight.ravel()))
            block(j,coords,weights)
            if rule=='matches':
                dest = (grids[j]+flows[j])[take]
                other = cameras[1-j]
                inside = ((dest>=0)&(dest<=[other.width-1,other.height-1])).all(axis=2)
                cycle = np.linalg.norm(flows[j][take]+sample(flows[1-j],dest),axis=2)
                block(1-j,dest,weights*inside*np.exp(-.5*(cycle/2.)**2))
        transform = ray_alignment(*(np.concatenate(p) for p in zip(*blocks)),affine=(rule in ('rays','matches')),similarity=(rule=='similarity'))
        diagnostics.append(None if transform is None else transform.tolist())
        if transform is None:
            continue
        for j,q in enumerate(maps):
            optical_axis = cameras[j].camera_to_world[:3,2] @ rotation
            depth = ((q@transform[:,:3].T + transform[:,3]) - positions[j]) @ optical_axis
            weight = np.where(np.isfinite(depth)&(depth>0),1.-1./np.maximum(confidence[j],1.),0.)
            predictions[j].append((np.where(weight>0,depth,0.),weight))
    depths = []
    for j,items in enumerate(predictions):
        if items:
            mass = np.sum([w for _,w in items],axis=0)
            numerator = np.sum([v*w for v,w in items],axis=0)
            depths.append(np.where(mass>0,numerator/np.maximum(mass,1e-12),np.nan))
        else:
            depths.append(np.full(grids[j].shape[:2],np.nan))
    return depths,diagnostics


def pointmap_renderer(images, cameras, flows, pair_maps, rule):
    from .stereo_transport import SurveyStereo

    if rule == 'median':
        depths = [pair_maps[j][0][j][..., 2] for j in range(2)]
        return SurveyStereo(images, cameras, flows, depths, 'median_stereo'), []
    if rule in ('source_fit','rays_rescaled'):
        depths,alignment=calibrated_pair_depth(pair_maps,cameras,None,'rays')
        raw=[pair_maps[j][0][j][...,2] for j in range(2)]
        return SurveyStereo(images,cameras,flows,list(zip(raw,depths)),rule),alignment
    depths, alignment = calibrated_pair_depth(pair_maps, cameras, flows, rule)
    return SurveyStereo(images, cameras, flows, depths, 'metric_pair'), alignment


def surface_charts(depths, cameras):
    """Sample the peer image using each source pixel's metric surface point."""
    charts = []
    for j,(depth,cam) in enumerate(zip(depths,cameras)):
        peer = cameras[1-j]
        xy = grid_for(cam)
        dirs = np.dstack((xy,np.ones(xy.shape[:2]))) @ np.linalg.inv(cam.intrinsics).T
        rays = dirs @ cam.camera_to_world[:3,:3].T
        # This is the same finite depth range used for metric priors in SurveyStereo.
        ground = -cam.camera_to_world[2,3]/rays[...,2]
        depth = np.where(np.isfinite(depth)&(depth>0),np.clip(depth,.15*ground,4.*ground),ground)
        world = cam.camera_to_world[:3,3] + depth[...,None]*rays
        local = (world-peer.camera_to_world[:3,3]) @ peer.camera_to_world[:3,:3]
        uvz = local @ peer.intrinsics.T
        uv = uvz[...,:2]/np.maximum(uvz[...,2:],1e-6)
        charts.append(np.where((local[...,2:]>0)&np.isfinite(uv),uv,-1e4).astype(np.float32))
    return charts


def rematch_surface(images, cameras, depths, flow_model):
    from .survey_flow import estimate_flow

    charts = surface_charts(depths,cameras)
    aligned = [cv2.remap(images[1-j],uv,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
               for j,uv in enumerate(charts)]
    offsets = estimate_flow(flow_model,images,aligned)
    flows, direct = [], []
    for cam,chart,offset in zip(cameras,charts,offsets):
        xy = grid_for(cam)
        refined = cv2.remap(chart,xy+offset,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        inside = ((xy+offset>=0)&(xy+offset<=[cam.width-1,cam.height-1])).all(-1)
        flows.append(np.where(inside[...,None],refined,-1e4)-xy)
        direct.append(chart-xy)
    return np.stack(flows),np.stack(direct)


def rematch_candidates(images,cameras,depths,raw,model):
    from .survey_flow import estimate_flow
    from .stereo_transport import qualify_flows, rectify_source_pair, unrectify_flow

    peers,h = rectify_source_pair(images,cameras)
    ground = unrectify_flow(estimate_flow(model,images,peers),cameras,h)
    surface,direct = rematch_surface(images,cameras,depths,model)
    ground_chosen,_ = qualify_flows(images,cameras,[ground,raw])
    surface_chosen,_ = qualify_flows(images,cameras,[surface,raw])
    return dict(surface_direct=direct,surface_refined=surface,
                ground_selected=ground_chosen,surface_selected=surface_chosen)
