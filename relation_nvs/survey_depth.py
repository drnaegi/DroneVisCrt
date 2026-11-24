"""Source-only monocular shape estimation for calibrated survey cameras."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_moge():
    sys.path[:0] = [str(ROOT/'code/moge'), str(ROOT/'code/utils3d')]
    from moge.model.v2 import MoGeModel
    checkpoint = torch.load(ROOT/'checkpoints/MoGe2/moge-2-vitb-normal.pt', map_location='cpu', weights_only=True)
    model = MoGeModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.cuda().eval()


@torch.inference_mode()
def infer_depth(model, image, camera):
    rgb = torch.from_numpy(image.copy()).permute(2,0,1).float().cuda()/255.
    fov = np.rad2deg(2*np.arctan(camera.width/(2*camera.intrinsics[0,0])))
    result = model.infer(rgb, fov_x=float(fov), resolution_level=3, use_fp16=True)
    depth = result['depth'].cpu().float().numpy()
    return np.where(result['mask'].cpu().numpy(), depth, np.nan)


