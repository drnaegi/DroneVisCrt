import torch
from clip.model import CLIP
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.encoders.modules import FrozenCLIPImageEmbedder


class LocalCLIPImageEmbedder(FrozenCLIPImageEmbedder):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model = CLIP(
            embed_dim=768, image_resolution=224, vision_layers=24,
            vision_width=1024, vision_patch_size=14, context_length=77,
            vocab_size=49408, transformer_width=768, transformer_heads=12,
            transformer_layers=12,
        )
        del self.model.transformer
        self.antialias = False
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)


class DeviceDDIMSampler(DDIMSampler):
    def register_buffer(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.to(self.model.device)
        setattr(self, name, value)
