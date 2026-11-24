"""Use disjoint source-image regions to select the pointmap shape and scale."""
import numpy as np

MODELS=('rays','scaled_self','scaled_rays','mixture')


def coefficients(y, own, ray):
    x=ray-own
    target=y-own
    slope=.5
    offset=np.median(target-slope*x)
    for _ in range(5):
        residual=target-offset-slope*x
        cutoff=max(.005,1.4826*np.median(np.abs(residual)))
        w=np.minimum(1.,cutoff/np.maximum(np.abs(residual),1e-10))
        xmean=np.average(x,weights=w)
        dx=x-xmean
        slope=np.clip(np.sum(w*dx*target)/max(np.sum(w*dx*dx),1e-12),0,1)
        offset=np.median(target-slope*x)
    return ((0.,1.),(np.median(target),0.),(np.median(y-ray),1.),(offset,slope))


def fit_pointmap_depth(plane, stereo, reliable, raw, rays, *, scale_only=False):
    rays_ok=np.isfinite(rays)&(rays>0)
    base=np.where(rays_ok,rays,plane)
    limit=lambda value:np.clip(value,.15*plane,4.*plane)
    if scale_only:
        anchors=rays_ok&reliable
        scale=float(np.median(stereo[anchors]/rays[anchors])) if anchors.any() else 1.
        return limit(np.where(rays_ok,base*scale,plane)), 'scaled_rays',{'scale':scale}
    own_ok=np.isfinite(raw)&(raw>0)
    yy,xx=np.indices(plane.shape)
    anchors=reliable&rays_ok&own_ok&(yy%4==0)&(xx%4==0)
    tile=(yy[anchors]//32+xx[anchors]//32)%2
    if np.bincount(tile,minlength=2).min()<64:
        return limit(base),'rays',{'anchors':int(anchors.sum())}
    y=np.log(stereo[anchors])
    own=np.log(raw[anchors])
    ray=np.log(rays[anchors])
    errors=np.zeros(len(MODELS))
    for fold in (0,1):
        fit=tile==fold
        keep=~fit
        for i,(offset,slope) in enumerate(coefficients(y[fit],own[fit],ray[fit])):
            estimate=offset+(1-slope)*own[keep]+slope*ray[keep]
            errors[i]+=.5*np.mean(np.minimum(np.abs(estimate-y[keep]),.25))
    i=int(np.argmin(errors))
    offset,slope=coefficients(y,own,ray)[i]
    # Retain available metric depth outside the joint support of the two pointmaps.
    valid=rays_ok&own_ok
    depth=base.copy()
    depth[valid]=np.exp(offset+(1-slope)*np.log(raw[valid])+slope*np.log(rays[valid]))
    return limit(depth),MODELS[i],dict(anchors=int(anchors.sum()),held_out=dict(zip(MODELS,errors.tolist())),
                                      offset=float(offset),ray_weight=float(slope))
