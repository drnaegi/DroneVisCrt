"""Bidirectional source matching for aerial view synthesis."""
import numpy as np


def get_flow_model():
    from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
    return raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).cuda().eval()


def estimate_flow(model,ims,partners=None):
    import torch
    import torch.nn.functional as f
    x=torch.tensor(np.stack(ims)).cuda().permute(0,3,1,2).float()/127.5-1
    h,w=x.shape[-2:]
    if min(h,w) < 128:
        raise ValueError('RAFT requires both raster dimensions to be at least 128 pixels')
    y = x.flip(0) if partners is None else torch.tensor(np.stack(partners)).cuda().permute(0,3,1,2).float()/127.5-1
    x,y = [f.pad(z,(0,(-w)%8,0,(-h)%8),mode='replicate') for z in (x,y)]
    with torch.inference_mode():
        flow=model(x,y,num_flow_updates=12)[-1]
    return flow[:,:,:h,:w].permute(0,2,3,1).cpu().numpy()


