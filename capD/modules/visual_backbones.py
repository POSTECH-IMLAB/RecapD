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
    def __init__(self, visual_feature_size: int, img_size: int):
        super().__init__(visual_feature_size)
        self.img_size = img_size
        arch = self._define_arch()
        depth = len(arch["in_features"])

        self.conv_img = nn.Conv2d(3, self.visual_feature_size, 1, 1)
        self.downblocks = nn.ModuleList([
            resD(
                arch["in_features"][i], arch["out_features"][i]
            ) for i in range(1, depth)
        ])
    
    def forward(self, x, **kwargs):
        out = self.conv_img(x)
        for dblock in self.downblocks:
            out = dblock(out) 
        return out

    def _define_arch(self):
        assert self.img_size in (64, 128, 256)
        if self.img_size == 256:
            in_features = (1, 2, 4, 8, 16, 16)
            out_features = (1, 2, 4, 8, 16, 16, 16)
        elif self.img_size == 128:
            in_features = (1, 2, 4, 8, 16)
            out_features = (1, 2, 4, 8, 16, 16)
        else:
            in_features = (1, 2, 4, 8)
            out_features = (1, 2, 4, 8, 16)

        return {
            "in_features": [3] + [self.visual_feature_size * i for i in in_features],
            "out_features":  [self.visual_feature_size * i for i in out_features]
        }


class TorchvisionVisualBackbone(VisualBackbone):
    r"""
    A visual backbone from `Torchvision model zoo
    <https://pytorch.org/docs/stable/torchvision/models.html>`_. Any model can
    be specified using corresponding method name from the model zoo.
    Args:
        name: Name of the model from Torchvision model zoo.
        visual_feature_size: Size of the channel dimension of output visual
            features from forward pass.
        pretrained: Whether to load ImageNet pretrained weights from Torchvision.
        frozen: Whether to keep all weights frozen during training.
    """

    def __init__(
        self,
        name: str = "resnet50",
        visual_feature_size: int = 2048,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__(visual_feature_size)

        self.cnn = getattr(torchvision.models, name)(
            pretrained, zero_init_residual=True
        )
        # Do nothing after the final residual stage.
        self.cnn.fc = nn.Identity()

        # Freeze all weights if specified.
        if frozen:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval()

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Compute visual features for a batch of input images.
        Args:
            image: Batch of input images. A tensor of shape ``(batch_size, 3,
                height, width)``.
        Returns:
            A tensor of shape ``(batch_size, channels, height, width)``, for
            example it will be ``(batch_size, 2048, 7, 7)`` for ResNet-50.
        """

        for idx, (name, layer) in enumerate(self.cnn.named_children()):
            out = layer(image) if idx == 0 else layer(out)

            # These are the spatial features we need.
            if name == "layer4":
                # shape: (batch_size, channels, height, width)
                return out

    def detectron2_backbone_state_dict(self) -> Dict[str, Any]:
        r"""
        Return state dict of visual backbone which can be loaded with
        `Detectron2 <https://github.com/facebookresearch/detectron2>`_.
        This is useful for downstream tasks based on Detectron2 (such as
        object detection and instance segmentation). This method renames
        certain parameters from Torchvision-style to Detectron2-style.
        Returns:
            A dict with three keys: ``{"model", "author", "matching_heuristics"}``.
            These are necessary keys for loading this state dict properly with
            Detectron2.
        """
        # Detectron2 backbones have slightly different module names, this mapping
        # lists substrings of module names required to be renamed for loading a
        # torchvision model into Detectron2.
        DETECTRON2_RENAME_MAPPING: Dict[str, str] = {
            "layer1": "res2",
            "layer2": "res3",
            "layer3": "res4",
            "layer4": "res5",
            "bn1": "conv1.norm",
            "bn2": "conv2.norm",
            "bn3": "conv3.norm",
            "downsample.0": "shortcut",
            "downsample.1": "shortcut.norm",
        }
        # Populate this dict by renaming module names.
        d2_backbone_dict: Dict[str, torch.Tensor] = {}

        for name, param in self.cnn.state_dict().items():
            for old, new in DETECTRON2_RENAME_MAPPING.items():
                name = name.replace(old, new)

            # First conv and bn module parameters are prefixed with "stem.".
            if not name.startswith("res"):
                name = f"stem.{name}"

            d2_backbone_dict[name] = param

        return {
            "model": d2_backbone_dict,
            "__author__": "Karan Desai",
            "matching_heuristics": True,
        }
