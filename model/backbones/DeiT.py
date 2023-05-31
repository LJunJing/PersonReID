from .ViT import ViT
import torch.nn as nn
import torch
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F
import numpy as np
from .vision_transformer import VisionTransformer
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches  #196
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches+1, self.embed_dim))  # [1,197,384]

    def forward(self, x, labels=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        """
        x:[32,3,256,128]  B*C*H*W
        """
        B = x.shape[0]  #32
        x = self.patch_embed(x)  # [32,128,768]
        pe = self.pos_embedding  # [1, 197,768]
        print("x",x.shape)
        print("pos_embed",pe.shape)
        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks: #每一个blk都是一个Transformer块（LN->MSA->LN->MLP)
            x = blk(x)

        x = self.norm(x) # x:[1,192,384]
        return x[:,0]

def deit_small_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        img_size=[256, 128], patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_small_patch16_224-cd65a155.pth')
        model.load_state_dict(ckpt['model'], strict=False)

    pe = model.pos_embedding[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    print("pe",pe.shape)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model
