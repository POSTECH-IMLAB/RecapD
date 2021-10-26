from typing import Any, Dict

import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F


class GeneratorBackbone(nn.Module):
    r"""
    Base class for all visual backbones. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.
    """

    def __init__(self, visual_feature_size: int, noise_size: int, img_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.noise_size = noise_size
        self.img_size = img_size

    def _define_arch(self):
        raise NotImplementedError

class G_Block(nn.Module):

    def __init__(self, in_features, out_features, cond_size, upsample):
        super(G_Block, self).__init__()

        self.learnable_sc = (in_features != out_features)
        self.upsample = upsample

        self.c1 = nn.Conv2d(in_features, out_features, 3, 1, 1)
        self.c2 = nn.Conv2d(out_features, out_features, 3, 1, 1)

        self.affine0 = affine(num_features=in_features, cond_size=cond_size)
        self.affine1 = affine(num_features=in_features, cond_size=cond_size)
        self.affine2 = affine(num_features=out_features, cond_size=cond_size)
        self.affine3 = affine(num_features=out_features, cond_size=cond_size)

        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_features, out_features, 1, stride=1, padding=0)

    def forward(self, x, c):
        out = self.shortcut(x) + self.gamma * self.residual(x, c)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)

        return out

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, c):
        h = self.affine0(x, c)
        F.leaky_relu_(h, 0.2)
        h = self.affine1(h, c)
        F.leaky_relu_(h, 0.2)
        h = self.c1(h)

        h = self.affine2(h, c)
        F.leaky_relu_(h, 0.2)
        h = self.affine3(h, c)
        F.leaky_relu_(h, 0.2)

        return self.c2(h)


class affine(nn.Module):

    def __init__(self, num_features, cond_size):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_size, cond_size)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cond_size, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_size, cond_size)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cond_size, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class DF_GEN(GeneratorBackbone):
    def __init__(self,
            visual_feature_size: int = 32,
            noise_size: int = 100,
            img_size: int = 128,
            cond_size: int = 256,
            **kwargs
        ):
        super().__init__(visual_feature_size, noise_size, img_size)
        arch = self._define_arch()
        depth = len(arch["in_features"])

        init_size = 4 * 4 * self.visual_feature_size * 8

        self.proj_noise = nn.Linear(self.noise_size, init_size)
        self.upblocks = nn.ModuleList([
            G_Block(in_features=arch["in_features"][i],
                    out_features=arch["out_features"][i],
                    cond_size=cond_size,
                    upsample=(i < (depth -1))
            ) for i in range(depth)
        ])

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(arch["out_features"][-1], 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, c, **kwargs):
        out = self.proj_noise(z)
        out = out.view(out.size(0), -1, 4, 4)

        for gblock in self.upblocks:
            out = gblock(out, c)

        out = self.conv_out(out)
        return out

    def _define_arch(self):
        assert self.img_size in (64, 128, 256)
        if self.img_size == 256:
            in_features= (8, 8, 8, 8, 8, 4, 2)
            out_features = (8, 8, 8, 8, 4, 2, 1)
        elif self.img_size == 128:
            in_features = (8, 8, 8, 8, 4, 2)
            out_features = (8, 8, 8, 4, 2, 1)
        else:
            in_features = (8, 8, 8, 4, 2)
            out_features = (8, 8, 4, 2, 1)

        return {
            'in_features': [i * self.visual_feature_size for i in in_features],
            'out_features': [i * self.visual_feature_size for i in out_features],
        }




class Style_GEN(GeneratorBackbone):
    def __init__(self,
        visual_feature_size: int = 32,
        noise_size: int = 100,
        img_size: int = 128,
        cond_size: int = 256,
        **kwargs
    ):

        super().__init__(visual_feature_size, noise_size, img_size)
        raise NotImplementedError

        
 