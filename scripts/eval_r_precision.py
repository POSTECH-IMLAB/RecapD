import os 
import sys



sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from typing import Any
from tqdm import tqdm
import numpy as np

from loguru import logger
import torch
from torch.utils.data import DataLoader
from PIL import Image

# fmt: off
from capD.config import Config
from capD.factories import (
    PretrainingDatasetFactory,GeneratorFactory,
    
)
from capD.utils.checkpointing import CheckpointManager
from capD.utils.common import common_parser, common_setup
import capD.utils.distributed as dist
from capD.utils.timer import Timer
from capD.modules.gan_loss import GANLoss
from capD.modules.embedding import CNN_ENCODER, RNN_ENCODER
from capD.models.captioning import VirTexModel 
from capD.metrics.metric_main import calc_metric

parser = common_parser(
    description="Train a capD model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default='exps/df_damsm_fa_rec/checkpoint_300000.pth',
    help="Path to a checkpoint to resume training from (if provided)."
)

# fmt: on

def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        device: Any = torch.device("cpu")
    else:
        device = torch.cuda.current_device()

    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER, SCHEDULER
    # -------------------------------------------------------------------------
    test_dataset = PretrainingDatasetFactory.from_config(_C, split="test")
    logger.info(f"Dataset size: {len(test_dataset)}")
    
    batch_size = _C.TRAIN.BATCH_SIZE 
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=test_dataset.collate_fn,
    )
    
    netG_ema = GeneratorFactory.from_config(_C).to(device)
    netG_ema.eval()

    # For text encoder
    if _C.TEXT_ENCODER.NAME == "damsm":
        text_encoder = RNN_ENCODER() 
        text_encoder.load_state_dict(torch.load("datasets/DAMSMencoders/coco/text_encoder100.pth", map_location="cpu"))
        text_encoder.to(device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    else:
        raise NotImplementedError

    if _A.resume_from is not None:
        start_iteration = CheckpointManager(
            netG_ema=netG_ema 
        ).load(_A.resume_from)
    else:
        raise NotImplementedError

    # Create tensorboard writer and checkpoint manager (only in master process).
    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    model = CNN_ENCODER().to(device).eval()
    save_dir = f"eval/{_A.resume_from.split('/')[1]}"
    metric = "damsm_r_precision"
    kwargs = {"metric":metric, "save_img":True, "save_dir":save_dir,
                "G":netG_ema, "text_encoder":text_encoder, "encoder":model,
                "data_loader":dataloader, "batch_size":_C.TRAIN.BATCH_SIZE, "device":device}

    with torch.no_grad():
        result_dict = calc_metric(**kwargs)
    logger.info(f"Eval metric, {metric}: {result_dict['results'][metric]}")
    


    
                
if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)

    # if _A.num_gpus_per_machine == 0:
    #     main(_A)
    # else:
    #     # This will launch `main` and set appropriate CUDA device (GPU ID) as
    #     # per process (accessed in the beginning of `main`).
    #     dist.launch(
    #         main,
    #         num_machines=_A.num_machines,
    #         num_gpus_per_machine=_A.num_gpus_per_machine,
    #         machine_rank=_A.machine_rank,
    #         dist_url=_A.dist_url,
    #         args=(_A, ),
    #     )
