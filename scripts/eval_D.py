from copy import deepcopy
import os 
import sys


sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from typing import Any

from loguru import logger
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, DistributedSampler
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
from capD.modules.gan_loss import GANLoss, magp, r1
from capD.modules.embedding import CNN_ENCODER, RNN_ENCODER
from capD.metrics.metric_main import calc_metric

from capD.modules.decoder import D_REC
import lpips

parser = common_parser(
    description="Train a capD model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default="exps/cond/checkpoint_307734.pth",
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
    train_dataset = PretrainingDatasetFactory.from_config(_C, split="train")
    test_dataset = PretrainingDatasetFactory.from_config(_C, split="test")
    logger.info(f"Dataset size: {len(train_dataset)}")
    # train_sampler = (
    #     DistributedSampler(train_dataset, shuffle=True)  # type: ignore
    #     if _A.num_gpus_per_machine > 0
    #     else None
    # )

    g = torch.Generator()
    g.manual_seed(_C.RANDOM_SEED) 

    train_sampler = None
    batch_size = _C.TRAIN.BATCH_SIZE // dist.get_world_size()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=_A.cpu_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=test_dataset.collate_fn,
    )
    
    
    netD = DiscriminatorFactory.from_config(_C)
    CheckpointManager(netD=netD).load(_A.resume_from)
    decoder = D_REC(H=_C.DISCRIMINATOR.LOGITOR.H)
    netD.to(device)
    netD.eval()
    netD.requires_grad_(False)
    decoder.to(device)

    opt_decoder = torch.optim.Adam(decoder.parameters(), lr=0.0001, betas=_C.OPTIM.D.BETAS)

    # Create an iterator from dataloader to sample batches perpetually.
    start_iteration = 0
    train_dataloader_iter = cycle(train_dataloader, device, start_iteration)

    # Keep track of time per iteration and ETA.
    timer = Timer(
        start_from=start_iteration + 1, total_iterations=10000
    )
    # Create tensorboard writer and checkpoint manager (only in master process).
    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    lpips_fn = lpips.LPIPS(net="vgg").cuda()
    lpips_fn.requires_grad_(False)

    for iteration in range(start_iteration + 1, 10000 + 1):
        timer.tic()
        decoder.train()
        batch = next(train_dataloader_iter)
        image = batch["image"].cuda()
        out_dict = netD(image)
        rec = decoder(out_dict["dec_features"])
        loss =lpips_fn(rec, image.detach()).mean()
        # Train Discriminator
        opt_decoder.zero_grad()
        loss.backward()
        opt_decoder.step()
        timer.toc()
        # ---------------------------------------------------------------------
        #   LOGGING
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        #  Checkpointing
        # ---------------------------------------------------------------------

    with torch.no_grad():
        val_loss = 0    
        decoder.eval()
        for data in test_dataloader:
            image = data["image"].cuda()
            out_dict = netD(image)
            rec = decoder(out_dict["dec_features"])
            loss = lpips_fn(rec, image).mean()
            val_loss += loss.item()

        print("val_loss: ", val_loss.mean())


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
