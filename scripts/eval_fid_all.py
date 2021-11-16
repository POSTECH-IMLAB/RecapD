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
from capD.modules.embedding import RNN_ENCODER
from capD.models.captioning import VirTexModel 

parser = common_parser(
    description="Train a capD model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
)
group.add_argument(
    "--checkpoint-every", type=int, default=4000,
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
    org_dir = f'eval/org128'
    save_org = False
    if not os.path.isdir(org_dir):
        os.makedirs(org_dir, exist_ok=True)
        save_org = True
    name = _A.resume_from.split('/')[1]
    save_dir = f'eval/{name}'
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    cond = False
    for batch in tqdm(dataloader):
        # Train Discriminator
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
            if save_org:
                org = batch["image"][j].data.cpu().numpy() 
                org = (org + 1.0) * 127.5
                org = org.astype(np.uint8)
                org = np.transpose(org, (1,2,0))
                org =  Image.fromarray(org)
                fullpath = os.path.join(org_dir, f'{batch["image_id"][j]}.png')
                org.save(fullpath)
            cnt += 1
            if cnt >= 30000:
                cond = True
                break
        if cond:
            break
                
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
