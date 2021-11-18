from copy import deepcopy
import os 
import sys


sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from typing import Any
import glob

from loguru import logger
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader#, DistributedSampler
import wandb
import torchvision.utils as vutils
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

# fmt: off
from capD.config import Config
from capD.factories import (
    PretrainingDatasetFactory, GeneratorFactory, DiscriminatorFactory, PretrainingModelFactory, TextEncoderFactory,
    
)
from capD.utils.checkpointing import CheckpointManager #, update_average
from capD.utils.common import common_parser, common_setup #, count_params, cycle, seed_worker
import capD.utils.distributed as dist
from capD.utils.timer import Timer
from capD.modules.embedding import CNN_ENCODER, RNN_ENCODER
from capD.metrics.metric_main import calc_metric

parser = common_parser(
    description="Train a capD model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-dir", default="exps/df_damsm",
    help="Path to a checkpoint to resume training from (if provided)."
)

group.add_argument(
    "--metrics", type=str, default="inception_fid30k_full"
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
    #if "inception" in _A.metrics:
    #    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    #    model = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0, verbose=True)
    

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
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    filenames = sorted(glob.glob(f"{_A.resume_dir}/checkpoint_*"), key=os.path.getmtime)[:79]

    if dist.is_master_process():
        wandb.init(project=f"{_A.prefix}_capD")
        wandb.config.update(_C)
        wandb.run.name = _A.config.split("configs/")[-1].split(".yaml")[0]
        wandb.run.save()

    for file in filenames:
        iteration = CheckpointManager(netG_ema=netG_ema).load(file)
        netG_ema.eval()

        org_dir = f'eval/org128'
        save_org = False
        if not os.path.isdir(org_dir):
            os.makedirs(org_dir, exist_ok=True)
            save_org = True
        name = _A.resume_dir.split('/')[1]
        save_dir = f'eval/{name}'
        os.makedirs(save_dir, exist_ok=True)
        cnt = 0
        stop_cond = False
        for batch in tqdm(test_dataloader):
            # Train Discriminator
            if save_org:
                for j in range(_C.TRAIN.BATCH_SIZE):
                    org = batch["image"][j].data.cpu().numpy() 
                    org = (org + 1.0) * 127.5
                    org = org.astype(np.uint8)
                    org = np.transpose(org, (1,2,0))
                    org =  Image.fromarray(org)
                    fullpath = os.path.join(org_dir, f'{batch["image_id"][j]}.png')
                    org.save(fullpath)
            if not stop_cond:
                with torch.no_grad():
                    tokens, tok_lens = batch["damsm_tokens"] ,batch["damsm_lengths"]
                    tokens, tok_lens = tokens.to(device), tok_lens.to(device)
                    hidden = text_encoder.init_hidden(tokens.size(0))
                    _, sent_embs = text_encoder(tokens, tok_lens, hidden)

                    z = torch.randn(_C.TRAIN.BATCH_SIZE, _C.GENERATOR.NOISE_SIZE).to(device)
                    fakes = netG_ema(z, sent_embs)

                for j in range(_C.TRAIN.BATCH_SIZE):
                    im = fakes[j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1,2,0))
                    im =  Image.fromarray(im)
                    fullpath = os.path.join(save_dir, f'{batch["image_id"][j]}.png')
                    im.save(fullpath)
                    cnt += 1
                    if cnt >= 30000:
                        stop_cond = True
                        break

            if not save_org and stop_cond:
                break
        
        if dist.is_master_process():
            org_stat = "eval/org128_stat.npz"
            with torch.no_grad():
                if not os.path.isfile(org_stat):
                    mu, sigma = compute_statistics_of_path(org_dir, model, 50, 2048, device, 4)
                    np.savez(org_stat, mu=mu, sigma=sigma)

                with np.load(org_stat) as f:
                    mu, sigma = f['mu'][:], f['sigma'][:]

                m, s = compute_statistics_of_path(save_dir, model, 50, 2048, device, 4)
                fid = calculate_frechet_distance(mu, sigma, m, s)
                logger.info(f"FID: {fid}")
                wandb.log({"fid":fid}, step=iteration)
                
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
