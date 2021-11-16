from typing import Any, Dict

import torch
from torch import nn
import torchvision

import torch.nn.functional as F


class VisualBackbone(nn.Module):
    r"""
    Base class for all visual backbones. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.
    """

    def __init__(self, visual_feature_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size

class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)

class DF_DISC(VisualBackbone):
    def __init__(self, visual_feature_size: int, img_size: int, H: int):
        super().__init__(visual_feature_size)
        self.img_size = img_size
        self.H = H
        arch = self._define_arch()
        depth = len(arch["in_features"])

        self.conv_img = nn.Conv2d(3, self.visual_feature_size, 3, 1, 1)
        self.downblocks = nn.ModuleList([
            resD(
                arch["in_features"][i], arch["out_features"][i]
            ) for i in range(1, depth)
        ])
    
    def forward(self, x, return_features = False, **kwargs):
        out = self.conv_img(x)
        for dblock in self.downblocks:
            out = dblock(out) 
            if out.size(-1) == self.img_size//16:
                dec = out
            if out.size(-1) == 8:
                out8 = out

        if not return_features:
            return out
        else:
            return out, dec, out8

    def _define_arch(self):
        assert self.img_size in (64, 128, 256)
        if self.img_size == 256:
            in_features = (1, 2, 4, 8, 16, 16)
            out_features = (1, 2, 4, 8, 16, 16)
        elif self.img_size == 128:
            in_features = (1, 2, 4, 8, 16)
            out_features = (1, 2, 4, 8, 16)
        else:
            in_features = (1, 2, 4, 16)
            out_features = (1, 2, 4, 16)

        return {
            "in_features": [3] + [self.visual_feature_size * i for i in in_features],
            "out_features":  [self.visual_feature_size * i for i in out_features] + [self.H],
        }
