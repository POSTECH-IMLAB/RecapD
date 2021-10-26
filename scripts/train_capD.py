import os 
import sys 

sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from collections import Counter
from itertools import accumulate
from typing import Any

from loguru import logger
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# fmt: off
from capD.config import Config
from capD.factories import (
    PretrainingDatasetFactory, OptimizerFactory, GeneratorFactory, DiscriminatorFactory, TextEncoderFactory,
    
)
from capD.utils.checkpointing import CheckpointManager
from capD.utils.common import common_parser, common_setup, cycle
import capD.utils.distributed as dist
from capD.utils.timer import Timer
from capD.modules.gan_loss import GANLoss


parser = common_parser(
    description="Train a capD model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
)
group.add_argument(
    "--checkpoint-every", type=int, default=2000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
group.add_argument(
    "--log-every", type=int, default=20,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device: Any = torch.device("cpu")
    else:
        # Get the current device as set for current distributed process.
        # Check `launch` function in `capD.utils.distributed` module.
        device = torch.cuda.current_device()

    # Create a config object (this will be immutable) and perform common setup
    # such as logging and setting up serialization directory.
    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER, SCHEDULER
    # -------------------------------------------------------------------------
    train_dataset = PretrainingDatasetFactory.from_config(_C, split="train")
    val_dataset = PretrainingDatasetFactory.from_config(_C, split="val")

    # Make `DistributedSampler`s to shard datasets across GPU processes.
    # Skip this if training on CPUs.
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)  # type: ignore
        if _A.num_gpus_per_machine > 0
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False)  # type: ignore
        if _A.num_gpus_per_machine > 0
        else None
    )

    batch_size = _C.TRAIN.BATCH_SIZE // dist.get_world_size()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )

    if _C.TEXT_ENCODER.FROZEN:
        assert not _C.OPTIM.G.UPDATE_EMB and not _C.OPTIM.D.UPDATE_EMB 
    else:
        assert _C.OPTIM.G.UPDATE_EMB or _C.OPTIM.D.UPDATE_EMB

    netG = GeneratorFactory.from_config(_C).to(device)
    netD = DiscriminatorFactory.from_config(_C).to(device)
    if _C.TEXT_ENCODER.NAME == "capD":
        text_encoder = netD.textual.embedding
    else:
        text_encoder = TextEncoderFactory.from_config(_C).to(device)

    gan_loss = GANLoss(_C.GAN_LOSS)

    g_param = []
    for name, param in netG.named_parameters():
        g_param.append((name, param))
    d_param = []
    for name, param in netD.named_parameters():
        d_param.append((name, param))
    
    if _C.TEXT_ENCODER.FROZEN:
        text_encoder.requires_grad_(False)
    else:
        if _C.OPTIM.G.UPDATE_EMB:
            for name, param in text_encoder.named_parameters():
                g_param.append((name, param))
        if _C.OPTIM.D.UPDATE_EMB:
            for name, param in text_encoder.named_parameters():
                d_param.append((name, param))
        
    optG = OptimizerFactory.from_config(_C.OPTIM.G, g_param)
    optD = OptimizerFactory.from_config(_C.OPTIM.D, d_param)

    # -------------------------------------------------------------------------
    #   BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------

    # Load checkpoint to resume training if specified.
    if _A.resume_from is not None:
        raise NotImplementedError
        start_iteration = CheckpointManager(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        ).load(_A.resume_from)
    else:
        start_iteration = 0

    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device, start_iteration)

    # Wrap model in DDP if using more than one processes.
    if dist.get_world_size() > 1:
        dist.synchronize()
        if not _C.TEXT_ENCODER.FROZEN:
            text_encoder = nn.parallel.DistributedDataParallel(
                text_encoder, device_ids=[device], find_unused_parameters=True
            )
        netG = nn.parallel.DistributedDataParallel(
            netG, device_ids=[device], find_unused_parameters=True
        )
        netD = nn.parallel.DistributedDataParallel(
            netD, device_ids=[device], find_unused_parameters=True
        )
    # Keep track of time per iteration and ETA.
    timer = Timer(
        start_from=start_iteration + 1, total_iterations=_C.TRAIN.NUM_ITERATIONS
    )
    # Create tensorboard writer and checkpoint manager (only in master process).
    if dist.is_master_process():
        tensorboard_writer = SummaryWriter(log_dir=_A.serialization_dir)
        tensorboard_writer.add_text("config", f"```\n{_C}\n```")

        checkpoint_manager = CheckpointManager(
            _A.serialization_dir,
            netG=netG,
            netD=netD,
            optG=optG,
            optD=optD,
            text_encoder = text_encoder if not _C.TEXT_ENCODER.FROZEN else None, 
        )

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, _C.TRAIN.NUM_ITERATIONS + 1):
        timer.tic()
        netG.train(), netD.train(), text_encoder.train()
        batch = next(train_dataloader_iter)

        # Train Discriminator
        z = torch.randn(_C.TRAIN.BATCH_SIZE, _C.GENERATOR.NOISE_SIZE).to(device)
        batch["z"] = z
        d_loss_dict = gan_loss.compute_d_loss(batch, text_encoder, netG, netD) 
        errD = gan_loss.accumulate_loss(d_loss_dict)

        optD.zero_grad(), optG.zero_grad()
        errD.backward()
        optD.step()

        # Train Generator
        g_loss_dict, fakes = gan_loss.compute_g_loss(batch, text_encoder, netG, netD)
        errG = gan_loss.accumulate_loss(g_loss_dict) 

        optD.zero_grad(), optG.zero_grad()
        errG.backward()
        optG.step()

        timer.toc()

        # ---------------------------------------------------------------------
        #   LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.log_every == 0:
            logger.info(
                f"{timer.stats} [errD {errD.detach():.3f} errG {errG.detach():.3f}] [GPU {dist.gpu_mem_usage()} MB]"
            )
            if dist.is_master_process():
                tensorboard_writer.add_image("fake", vutils.make_grid(fakes.detach(), normalize=True, scale_each=True), iteration)
                tensorboard_writer.add_image("real", vutils.make_grid(batch["image"], normalize=True, scale_each=True), iteration)
                tensorboard_writer.add_scalars("D", d_loss_dict, iteration)
                tensorboard_writer.add_scalars("G", g_loss_dict, iteration)
                
                #raise NotImplementedError 
                # wandb
                # tensorboard_writer.add_scalars(
                #     "learning_rate",
                #     {
                #         "visual": optimizer.param_groups[0]["lr"],
                #         "common": optimizer.param_groups[-1]["lr"],
                #     },
                #     iteration,
                # )
                # tensorboard_writer.add_scalars(
                #     "train", output_dict["loss_components"], iteration
                # )

        # ---------------------------------------------------------------------
        #   VALIDATION
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            if dist.is_master_process():
                checkpoint_manager.step(iteration)

            # All processes will wait till master process is done serializing.
            dist.synchronize()

            # torch.set_grad_enabled(False)
            # text_encoder.eval(), netG.eval()

            # word_embeddings = text_encoder(batch["caption_tokens"])
            # sent_embeddings = torch.sum(word_embeddings, dim=1)
            # sent_embeddings = sent_embeddings / batch["caption_lengths"].unsqueeze(1)
            # # normalize
            # sent_embeddings = sent_embeddings * (sent_embeddings.square().mean(1, keepdim=True) + 1e-8).rsqrt()

            # fakes = netG(batch["z"], sent_embeddings) 
            #vutils.save_image(fakes.data, os.path.join(_A.serialization_dir, f'{iteration}.png'), normalize=True, scale_each=True, nrow=8)
            #torch.set_grad_enabled(True)
            # if dist.is_master_process():
            #     tensorboard_writer.add_scalars("val", val_loss_dict, iteration)


if __name__ == "__main__":
    _A = parser.parse_args()

    if _A.num_gpus_per_machine == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus_per_machine,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            args=(_A, ),
        )
