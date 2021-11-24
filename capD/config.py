from typing import Any, List, Optional

from fvcore.common.config import CfgNode as CN


class Config(object):
    r"""
    This class provides package-wide configuration management. It is a
    nested dict-like structure with nested keys accessible as attributes. It
    contains sensible default values, which can be modified by (first) a YAML
    file and (second) a list of attributes and values.
    An instantiated object is immutable: modifying any attribute is illegal.
    You must override required parameter values either through ``config_file``
    or ``override_list`` arguments.
    Args:
        config_file: Path to a YAML file containing config parameters.
        config_override: A list of sequential attributes and values of parameters.
            This happens after overriding from YAML file.
    Examples:
        Let a YAML file named "config.yaml" specify these parameters to override::
            OPTIM:
            BATCH_SIZE: 512
            LR: 0.01
        >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 1024])
        >>> _C.LR  # default: 0.001
        0.01
        >>> _C.OPTIM.BATCH_SIZE  # default: 256, file: 512
        1024
    """

    def __init__(
        self, config_file: Optional[str] = None, override_list: List[Any] = []
    ):
        _C = CN()

        _C.RANDOM_SEED = 0
        _C.AMP = False
        _C.CUDNN_DETERMINISTIC = False
        _C.CUDNN_BENCHMARK = True

        # ---------------------------------------------------------------------
        #   Data paths and parameters related to dataloading.
        # ---------------------------------------------------------------------
        _C.DATA = CN()

        _C.DATA.ROOT = "datasets/coco"
        _C.DATA.TOKENIZER_MODEL = "datasets/vocab/coco17_10k.model"

        _C.DATA.VOCAB_SIZE = 10000 # 27297 for damsm
        _C.DATA.UNK_INDEX = 0
        _C.DATA.SOS_INDEX = 1
        _C.DATA.EOS_INDEX = 2
        _C.DATA.MASK_INDEX = 3

        _C.DATA.IMAGE_CROP_SIZE = 128 
        _C.DATA.MAX_CAPTION_LENGTH = 30

        _C.DATA.CAPTION_PER_IMAGE = 5

        _C.DATA.IMAGE_TRANSFORM_TRAIN = [
            "random_resized_crop",
            "horizontal_flip",
            #"color_jitter",
            "normalize",
        ]

        _C.DATA.IMAGE_TRANSFORM_TEST = [
            "smallest_resize",
            "center_crop",
            "normalize",
        ]

        # ---------------------------------------------------------------------
        #   Model architecture: visual backbone and textual head.
        # ---------------------------------------------------------------------
        _C.TEXT_ENCODER = CN()
        _C.TEXT_ENCODER.NAME = "damsm" # "damsm" "random", "capD", "virtex"
        _C.TEXT_ENCODER.DIR = "datasets/DAMSMencoders/text_encoder100.pth"
        _C.TEXT_ENCODER.EMBEDDING_SIZE = 256
        _C.TEXT_ENCODER.FROZEN = True 

        _C.GENERATOR = CN()
        _C.GENERATOR.NAME = "df"
        _C.GENERATOR.NOISE_SIZE = 100
        _C.GENERATOR.FEATURE_SIZE = 32
        
        _C.DISCRIMINATOR = CN()
        _C.DISCRIMINATOR.NAME = "df"
        _C.DISCRIMINATOR.VISUAL = CN()
        _C.DISCRIMINATOR.VISUAL.NAME = "df" #"torchvision::resnet50" # "df"
        _C.DISCRIMINATOR.VISUAL.FEATURE_SIZE = 512 #2048
        _C.DISCRIMINATOR.VISUAL.PRETRAINED = False 
        _C.DISCRIMINATOR.VISUAL.FROZEN = False
        _C.DISCRIMINATOR.VISUAL.DECODER = False 

        _C.DISCRIMINATOR.LOGITOR = CN()
        _C.DISCRIMINATOR.LOGITOR.NAME = "df"
        _C.DISCRIMINATOR.LOGITOR.H = 512 #2048

        _C.DISCRIMINATOR.TEXTUAL = CN()
        _C.DISCRIMINATOR.TEXTUAL.NAME = "transdec_postnorm::L1_H512_A8_F2048"
        _C.DISCRIMINATOR.TEXTUAL.CAPTION_BACKWARD = True
        _C.DISCRIMINATOR.TEXTUAL.DROPOUT = 0.1
        _C.DISCRIMINATOR.TEXTUAL.PRETRAINED = False 
        _C.DISCRIMINATOR.TEXTUAL.FROZEN = False 

        _C.DISCRIMINATOR.TEXTUAL.DECODER = CN()
        _C.DISCRIMINATOR.TEXTUAL.DECODER.NAME = "beam_search"
        _C.DISCRIMINATOR.TEXTUAL.DECODER.BEAM_SIZE = 5
        _C.DISCRIMINATOR.TEXTUAL.DECODER.NUCLEUS_SIZE = 0.9
        _C.DISCRIMINATOR.TEXTUAL.DECODER.MAX_DECODING_STEPS = _C.DATA.MAX_CAPTION_LENGTH

        # ---------------------------------------------------------------------
        #   Optimization hyper-parameters, default values are for pretraining
        #   our best model on bicaptioning task (COCO Captions).
        # ---------------------------------------------------------------------

        _C.TRAIN = CN()
        _C.TRAIN.BATCH_SIZE = 32 
        _C.TRAIN.NUM_ITERATIONS = 300000

        _C.GAN_LOSS = CN()
        _C.GAN_LOSS.TYPE = "hinge"
        _C.GAN_LOSS.D_LOSS_COMPONENT = "logit,magp"
        _C.GAN_LOSS.G_LOSS_COMPONENT = "logit"
        _C.GAN_LOSS.GP = True 
        _C.GAN_LOSS.LOGIT_INPUT = "visual_features"
        _C.GAN_LOSS.FA_FEATURE = "visual_features"
        _C.GAN_LOSS.LOGIT_STOP_GRAD = False
        _C.GAN_LOSS.CAP_STOP_GRAD = False
        _C.GAN_LOSS.SLOW_CAPG = False
        _C.GAN_LOSS.CAP_COEFF = 1.
        
        _C.OPTIM = CN()
        _C.OPTIM.G = CN()
        _C.OPTIM.D = CN()

        _C.OPTIM.G.OPTIMIZER_NAME = "adam"
        _C.OPTIM.G.VISUAL_LR = 0.0001
        _C.OPTIM.G.TEXT_LR = 0.0001
        _C.OPTIM.G.BETAS = [0.0, 0.9]
        _C.OPTIM.G.SGD_MOMENTUM = 0.9
        _C.OPTIM.G.WIEGHT_DECAY = 0.0001
        _C.OPTIM.G.NO_DECAY = ""
        _C.OPTIM.G.CLIP_GRAD_NORM = 0.0
        _C.OPTIM.G.LOOKAHEAD = CN()
        _C.OPTIM.G.LOOKAHEAD.USE = False 
        _C.OPTIM.G.LOOKAHEAD.ALPHA = 0.5
        _C.OPTIM.G.LOOKAHEAD.STEPS = 5
        _C.OPTIM.G.WARMUP_STEPS = 10000
        _C.OPTIM.G.LR_DECAY_NAME = "cosine"
        _C.OPTIM.G.LR_STEPS = []
        _C.OPTIM.G.LR_GAMMA = 0.1
        _C.OPTIM.G.UPDATE_EMB = False


        _C.OPTIM.D.OPTIMIZER_NAME = "adam" #"sgd"
        _C.OPTIM.D.VISUAL_LR = 0.0004
        _C.OPTIM.D.TEXT_LR = 0.0004
        _C.OPTIM.D.BETAS = [0.0, 0.9]
        _C.OPTIM.D.SGD_MOMENTUM = 0.9
        _C.OPTIM.D.WEIGHT_DECAY = 0.0001
        _C.OPTIM.D.NO_DECAY = ".*textual.(embedding|transformer).*(norm.*|bias)"
        _C.OPTIM.D.CLIP_GRAD_NORM = 0.0
        _C.OPTIM.D.LOOKAHEAD = CN()
        _C.OPTIM.D.LOOKAHEAD.USE = False 
        _C.OPTIM.D.LOOKAHEAD.ALPHA = 0.5
        _C.OPTIM.D.LOOKAHEAD.STEPS = 5
        _C.OPTIM.D.WARMUP_STEPS = 10000
        _C.OPTIM.D.LR_DECAY_NAME = "cosine"
        _C.OPTIM.D.LR_STEPS = []
        _C.OPTIM.D.LR_GAMMA = 0.1
        _C.OPTIM.D.UPDATE_EMB = False

        # Override parameter values from YAML file first, then from override
        # list, then add derived params.
        self._C = _C
        if config_file is not None:
            self._C.merge_from_file(config_file)
        self._C.merge_from_list(override_list)

        self.add_derived_params()

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def add_derived_params(self):
        r"""Add parameters with values derived from existing parameters."""

        # We don't have any such cases so far.
        pass

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        return self._C.__str__()

    def __repr__(self):
        return self._C.__repr__()
