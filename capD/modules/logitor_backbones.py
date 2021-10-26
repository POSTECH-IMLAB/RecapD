from typing import Any, Dict

import torch
from torch import nn


class LogitorBackbone(nn.Module):

    def __init__(self, visual_feature_size: int ):
        super().__init__()
        self.visual_feature_size = visual_feature_size 

class DF_LOGIT(LogitorBackbone):
    def __init__(self, visual_feature_size:int = 32, cond_size:int = 256, **kwargs):
        super().__init__(visual_feature_size)

        self.conv_joint = nn.Sequential(
            nn.Conv2d(16 * visual_feature_size + cond_size, 2 * visual_feature_size, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * visual_feature_size, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x, c):
        c = c.view(c.size(0), -1 , 1, 1)
        c = c.repeat(1, 1, c.size(-2),c.size(-1))
        x_c_code = torch.cat((x, c), dim=1)
        out = self.conv_joint(x_c_code)
        return out


#class PROJ_LOGIT
