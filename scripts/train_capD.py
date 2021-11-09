from copy import deepcopy
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
    PretrainingDatasetFactory, OptimizerFactory, GeneratorFactory, DiscriminatorFactory, PretrainingModelFactory, TextEncoderFactory,
    
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
    "--resume-from", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
)
group.add_argument(
    "--checkpoint-every", type=int, default=2500, #default=4000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
group.add_argument(
    "--log-every", type=int, default=20,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
group.add_argument(
    "--metrics", type=str, default="damsm_r_precision"
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
    
    
    netG = GeneratorFactory.from_config(_C).to(device)
    netG_ema = deepcopy(netG).eval().requires_grad_(False)
    netD = DiscriminatorFactory.from_config(_C)

    # For loading pretrained model
    if _C.DISCRIMINATOR.TEXTUAL.PRETRAINED or _C.TEXT_ENCODER == "virtex":
        _V = Config("configs/bicaptioning_R_50_L1_H2048.yaml")
        model = PretrainingModelFactory.from_config(_V) 
        CheckpointManager(model=model).load("bicaptioning_R_50_L1_H2048.pth")

    if  _C.DISCRIMINATOR.TEXTUAL.PRETRAINED:
        text_dict = model.textual.state_dict()
        del text_dict["visual_projection.weight"]
        del text_dict["visual_projection.bias"]
        netD.textual.load_state_dict(text_dict, strict=False)
    if _C.DISCRIMINATOR.VISUAL.PRETRAINED:
        netD.visual.load_state_dict(model.visual.state_dict())
        if _C.DISCRIMINAOTR.VISUAL.FROZEN:
            netD.visual.requires_grad_(False)
            netD.visual.eval()
    netD.to(device)

    logger.info(f"netG param: {count_params(netG)}")
    logger.info(f"netD param: {count_params(netD)}")

    # For text encoder
    if _C.TEXT_ENCODER.FROZEN:
        assert not _C.OPTIM.G.UPDATE_EMB and not _C.OPTIM.D.UPDATE_EMB 
    else:
        assert _C.OPTIM.G.UPDATE_EMB or _C.OPTIM.D.UPDATE_EMB

    if _C.TEXT_ENCODER.NAME == "capD":
        text_encoder = netD.textual.embedding
    elif _C.TEXT_ENCODER.NAME == "virtex":
        text_encoder = model.textual.embedding
    elif _C.TEXT_ENCODER.NAME == "damsm":
        text_encoder = RNN_ENCODER().to(device)
    else:
        text_encoder = TextEncoderFactory.from_config(_C).to(device)

    if _C.TEXT_ENCODER.FROZEN:
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    gan_loss = GANLoss(_C.GAN_LOSS)

    g_param_group = []
    for name, param in netG.named_parameters():
        if param.requires_grad:
            g_param_group.append({"params":[param], "lr":_C.OPTIM.G.VISUAL_LR})

    d_param_group = []
    if not _C.OPTIM.D.UPDATE_EMB:
        text_encoder.requires_grad_(False)
    for name, param in netD.named_parameters():
        if param.requires_grad:
            lr = _C.OPTIM.D.TEXT_LR if "textual" in name else _C.OPTIM.D.VISUAL_LR
            d_param_group.append({"params":[param], "lr":lr})
    
    if _C.OPTIM.G.UPDATE_EMB:
        text_encoder.requires_grad_(True)
        for name, param in text_encoder.named_parameters():
            if param.requires_grad:
                g_param_group.append({"params":[param], "lr":_C.OPTIM.G.TEXT_LR})
        
    optG = torch.optim.Adam(g_param_group, betas=_C.OPTIM.G.BETAS)
    optD = torch.optim.Adam(d_param_group, betas=_C.OPTIM.D.BETAS)
    # -------------------------------------------------------------------------
    #   BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------

    # Load checkpoint to resume training if specified.
    if _A.resume_from is not None:
        start_iteration = CheckpointManager(
            netG=netG, netD=netD, optG=optG, optD=optD, netG_ema=netG_ema,
            text_encoder=text_encoder if not _C.TEXT_ENCODER.FROZEN else None,
        ).load(_A.resume_from)
    else:
        start_iteration = 0

    # Create an iterator from dataloader to sample batches perpetually.
    train_dataloader_iter = cycle(train_dataloader, device, start_iteration)

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
            netG_ema = netG_ema,
            text_encoder = text_encoder if not _C.TEXT_ENCODER.FROZEN else None, 
        )

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, _C.TRAIN.NUM_ITERATIONS + 1):
        timer.tic()
        netG.train(), netD.train(), 
        if _C.TEXT_ENCODER.FROZEN:
            text_encoder.eval() 

        batch = next(train_dataloader_iter)
        # Train Discriminator
        z = torch.randn(_C.TRAIN.BATCH_SIZE, _C.GENERATOR.NOISE_SIZE).to(device)
        batch["z"] = z
        d_loss_dict, rec, cap_real = gan_loss.compute_d_loss(batch, text_encoder, netG, netD) 
        errD = gan_loss.accumulate_loss(d_loss_dict)

        optD.zero_grad(), optG.zero_grad()
        errD.backward()
        if _C.OPTIM.D.CLIP_GRAD_NORM > 1.0:
            torch.nn.utils.clip_grad_norm_(netD.parameters(), _C.OPTIM.D.CLIP_GRAD_NORM)
        optD.step()

        gp_loss_dict = gan_loss.compute_gp(batch, text_encoder, netD)
        errD_reg = gan_loss.accumulate_loss(gp_loss_dict)

        optD.zero_grad(), optG.zero_grad()
        errD_reg.backward()
        if _C.OPTIM.D.CLIP_GRAD_NORM > 1.0:
            torch.nn.utils.clip_grad_norm_(netD.parameters(), _C.OPTIM.D.CLIP_GRAD_NORM)
        optD.step()

        # Train Generator
        g_loss_dict, fakes, cap_fake = gan_loss.compute_g_loss(batch, text_encoder, netG, netD)
        if _C.GAN_LOSS.SLOW_CAPG:
            if g_loss_dict["errG_cap"] < d_loss_dict["errD_cap"]:
                del g_loss_dict["errG_cap"]
        errG = gan_loss.accumulate_loss(g_loss_dict) 

        optD.zero_grad(), optG.zero_grad()
        errG.backward()
        # todo add text encoder
        if _C.OPTIM.G.CLIP_GRAD_NORM > 1.0:
            torch.nn.utils.clip_grad_norm_(netG.parameters(), _C.OPTIM.G.CLIP_GRAD_NORM)
        optG.step()
        update_average(netG, netG_ema)
    
        timer.toc()

        # ---------------------------------------------------------------------
        #   LOGGING
        # ---------------------------------------------------------------------
        if iteration % _A.log_every == 0:
            log = f"{timer.stats} [errD {errD.detach():.3f} errG {errG.detach():.3f}] [GPU {dist.gpu_mem_usage()} MB]\n"
            for key in d_loss_dict:
                log += f'{key}: {d_loss_dict[key].detach():.3f} '
            for key in g_loss_dict:
                log += f'{key}: {g_loss_dict[key].detach():.3f} '
            logger.info(log)
            if _A.config == "configs/debug.yaml":
                vutils.save_image(fakes.data, f'fake.png', normalize=True, scale_each=True)
                vutils.save_image(batch["image"].data, f"real.png", normalize=True, scale_each=True)
                if rec is not None:
                    vutils.save_image(rec.data, f'rec.png', normalize=True, scale_each=True)
            if dist.is_master_process():
                tensorboard_writer.add_scalars("D", d_loss_dict, iteration)
                tensorboard_writer.add_scalars("G", g_loss_dict, iteration)

        # ---------------------------------------------------------------------
        #  Checkpointing
        # ---------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0 or iteration == _C.TRAIN.NUM_ITERATIONS:
            if dist.is_master_process():
                checkpoint_manager.step(iteration)
                checkpoint_manager.step(-1)

                netG_ema.eval(), netD.eval()
                text_iter = iter(test_dataloader)
                test_batch = next(text_iter)
                with torch.no_grad():
                    for key in test_batch:
                        if isinstance(test_batch[key], torch.Tensor):
                            test_batch[key] = test_batch[key].to(device)
                    tokens, tok_lens = test_batch["damsm_tokens"], test_batch["damsm_lengths"]
                    hidden = text_encoder.init_hidden(tokens.size(0))
                    _, sent_embs = text_encoder(tokens, tok_lens, hidden)
                    real_dict = netD(test_batch["image"])
                    fake_imgs = netG_ema(z, sent_embs)
                    fake_dict = netD(fake_imgs)
                    vutils.save_image(fake_imgs.data, os.path.join(_A.serialization_dir, f'{iteration}.png'), normalize=True, scale_each=True, nrow=8)
                    vutils.save_image(test_batch["image"].data, os.path.join(_A.serialization_dir, f"real.png"), normalize=True, scale_each=True, nrow=8)                

                    if "cap" in _C.GAN_LOSS.D_LOSS_COMPONENT:
                        org = test_dataset.tokenizer.decode(test_batch['caption_tokens'][0].tolist())
                        real = test_dataset.tokenizer.decode(real_dict['predictions'][0].tolist())
                        fake = test_dataset.tokenizer.decode(fake_dict['predictions'][0].tolist())
                        logger.info(f"org: {org}")
                        logger.info(f"real: {real}")
                        logger.info(f"fake: {fake}")
                        tensorboard_writer.add_text("captioning",f"{org} \n\n{real} \n\n{fake}")

                    for metric in metrics:
                        kwargs = {"metric": metric, "G":netG_ema, "text_encoder":text_encoder, "data_loader": test_dataloader, "batch_size":batch_size, "device":device}
                        if "clip" or "damsm" in metric:
                            kwargs["encoder"] = model
                        result_dict = calc_metric(**kwargs)
                        for key in result_dict["results"]:
                            logger.info(f"Eval metrics, {key}: {result_dict['results'][key]}")
                            tensorboard_writer.add_scalar(key, result_dict["results"][key], iteration)

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
