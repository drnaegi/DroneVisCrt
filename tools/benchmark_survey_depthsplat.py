"""Use the official DL3DV DepthSplat two-view model at its 256x448 input size."""
import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from relation_nvs.io import sha256
from tools.benchmark_survey_mvsplat import register_upstream, generate, ROOT


def run(args):
    if sha256(args.checkpoint) != '02c7b19d8d038f71b512ef434d42b49698c3288d8ed65b748c04bd43d298f37a':
        raise ValueError('Unexpected DepthSplat weights')
    register_upstream(args.upstream)
    from src.model.encoder.encoder_depthsplat import EncoderDepthSplat
    from src.model.decoder.cuda_splatting import render_cuda

    config = OmegaConf.load(args.upstream/'config/experiment/dl3dv.yaml')
    encoder_config = OmegaConf.merge(OmegaConf.load(args.upstream/'config/model/encoder/depthsplat.yaml'),
                                     config.model.encoder, dict(num_scales=2, upsample_factor=4,
                                     lowest_feature_resolution=8, monodepth_vit_type='vitb',
                                     supervise_intermediate_depth=False))
    original_load = torch.hub.load
    def local_backbone(repo, name):
        if (repo,name) != ('facebookresearch/dinov2','dinov2_vitb14'):
            raise ValueError((repo,name))
        return original_load(str(args.dinov2), name, source='local', pretrained=False)
    with patch.object(torch.hub, 'load', local_backbone):
        model = EncoderDepthSplat(encoder_config)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}, strict=True)
    model = model.cuda().eval()
    for group in args.groups:
        for composite in (False,True):
            generate(group, model, render_cuda, config.dataset, '256x448', composite, 'depthsplat_dl3dv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--dinov2',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/DepthSplat/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth')
    parser.add_argument('groups',nargs='+')
    run(parser.parse_args())
