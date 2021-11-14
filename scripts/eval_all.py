from copy import deepcopy
import os 
import sys


sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from collections import Counter
from itertools import accumulate
from typing import Any
import glob

from loguru import logger
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, DistributedSampler
#from torch.utils.tensorboard import SummaryWriter
import wandb
import torchvision.utils as vutils

# fmt: off
from capD.config import Config
from capD.factories import (
    PretrainingDatasetFactory, GeneratorFactory, DiscriminatorFactory, PretrainingModelFactory, TextEncoderFactory,
    
)
from capD.utils.checkpointing import CheckpointManager, update_average
from capD.utils.common import common_parser, common_setup, count_params, cycle, seed_worker
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
    "--resume-dir", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
)

group.add_argument(
    "--metrics", type=str, default="damsm_r_precision"
)
group.add_argument(
    "--prefix", type=str, default="local" 
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        device: Any = torch.device("cpu")
    else:
        device = torch.cuda.current_device()

    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)
    metrics = _A.metrics.split(',')

    if "clip" in _A.metrics:
        import clip 
        model, preprocess = clip.load("ViT-B/32", device=device)
    if "damsm" in _A.metrics:
        model = CNN_ENCODER().to(device)

    test_dataset = PretrainingDatasetFactory.from_config(_C, split="test")
    logger.info(f"Dataset size: {len(test_dataset)}")

    g = torch.Generator()
    g.manual_seed(_C.RANDOM_SEED) 

    batch_size = _C.TRAIN.BATCH_SIZE // dist.get_world_size()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=test_dataset.collate_fn,
    )
    
    netG_ema = GeneratorFactory.from_config(_C).to(device)
    text_encoder = RNN_ENCODER().to(device)
    filenames = glob.glob(f"{_A.resume_dir}/checkpoint_*") 

    if dist.is_master_process():
        wandb.init(project=f"{_A.prefix}_capD")
        wandb.config.update(_C)
        wandb.run.name = _A.config.split("configs/")[-1].split(".yaml")[0]
        wandb.run.save()

    for file in filenames:
        iteration = CheckpointManager(netG_ema=netG_ema).load(_A.resume_from)
        netG_ema.eval()

        if dist.is_master_process():
            with torch.no_grad():
                for metric in metrics:
                    kwargs = {"metric": metric, "G":netG_ema, "text_encoder":text_encoder, "data_loader": test_dataloader, "batch_size":batch_size, "device":device}
                    if "clip" or "damsm" in metric:
                        kwargs["encoder"] = model
                    result_dict = calc_metric(**kwargs)
                    for key in result_dict["results"]:
                        logger.info(f"Eval metrics, {key}: {result_dict['results'][key].detach()}")
                        wandb.log({key: result_dict["results"][key]}, step=iteration)
                        #tensorboard_writer.add_scalar(key, result_dict["results"][key], iteration)

            # if dist.is_master_process():
            #     tensorboard_writer.add_scalars("val", val_loss_dict, iteration)


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
