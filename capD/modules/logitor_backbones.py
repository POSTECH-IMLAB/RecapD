from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F


class LogitorBackbone(nn.Module):

    def __init__(self, H: int, cond_size: int = 0, contra: bool = False):
        super().__init__()
        self.H = H 
        self.cond_size = cond_size
        self.contra = contra
        self.proj_cond = nn.Linear(cond_size, 256) if (cond_size != 256 and cond_size != 0) else nn.Identity()
        if self.contra:
            self.proj_img = nn.Linear(H, 256)

    def get_contra_img_feat(self, visual_features):
        out = F.adaptive_avg_pool2d(visual_features, (1,1))
        out = out.view(out.size(0), -1)
        out = self.proj_img(out)
        return out

    def get_contra_sent_feat(self, sent_embs):
        out = self.proj_cond(sent_embs)
        return out

class DF_COND_LOGIT(LogitorBackbone):
    def __init__(self, H:int = 32*16, cond_size:int = 256, contra:bool = False, **kwargs):
        super().__init__(H, cond_size, contra)
        self.conv_joint = nn.Sequential(
            nn.Conv2d(H + 256, H//8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(H//8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x, c):
        c = self.proj_cond(c)
        c = c.view(c.size(0), -1 , 1, 1)
        c = c.repeat(1, 1, x.size(-2), x.size(-1))
        x_c_code = torch.cat((x, c), dim=1)
        out = self.conv_joint(x_c_code)
        return out

class DF_UNCOND_LOGIT(LogitorBackbone):
    def __init__(self, H:int = 32*16, **kwargs):
        super().__init__(H)
        self.conv_joint = nn.Sequential(
            nn.Conv2d(H, H//8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(H//8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x, **kwargs):
        out = self.conv_joint(x)
        return out
