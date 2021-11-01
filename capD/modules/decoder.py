import torch 
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes * 2,3,1,1),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# todo: 256x256
class D_REC(nn.Module):
    def __init__(self, H=2048):
        super().__init__()
        self.block0 = upBlock(H, H//8)
        self.block1 = upBlock(H//8, H//16)
        self.block2 = upBlock(H//16, H//32)
        self.block3 = upBlock(H//32, H//64)
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(H//64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x8):
        out = self.block0(x8)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.conv_out(out)
        return out
