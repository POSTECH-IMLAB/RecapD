# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Precision/Recall (PR) from the paper "Improved Precision and Recall
Metric for Assessing Generative Models". Matches the original implementation
by Kynkaanniemi et al. at
https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py"""

import os
import numpy as np
from PIL import Image
from genericpath import exists
from tqdm import tqdm
import torch
from . import metric_utils
from ..utils.metrics import truncated_z_sample

#----------------------------------------------------------------------------

def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

#----------------------------------------------------------------------------

def compute_damsm_r_precision(opts, num_gen=10000, R=1, r=100):
    assert num_gen % r == 0
    model = opts.encoder
    data_iter = iter(opts.data_loader)
    correct = []
    results = dict()

    for _ in tqdm(range(num_gen // r)):
        r_count = 0
        image_features = []
        text_features = []    
        keys = []
        # 1: gt, 99: mismatch 
        while r_count < r:
            images = []
            batch = next(data_iter)
            batch_size = opts.batch_size #batch["image"].size(0)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(opts.device)
            tokens, tok_lens = batch["damsm_tokens"], batch["damsm_lengths"]
            hidden = opts.text_encoder.init_hidden(tokens.size(0))
            _, sent_embs = opts.text_encoder(tokens, tok_lens, hidden)
            #z = torch.randn([batch_size, opts.G.noise_size], device=opts.device)
            z = truncated_z_sample(batch_size, opts.G.noise_size, seed=100)
            z = torch.from_numpy(z).float().to(opts.device)
            img = opts.G(z, sent_embs)
            #img = batch["image"]
            images.append(img)
            images = torch.cat(images)
            _, img_feat = model(images)
            image_features.append(img_feat)
            text_features.append(sent_embs)
            if opts.save_img:
                keys.append(batch["image_id"])
            r_count += batch_size
            if opts.save_img:
                os.makedirs(opts.save_dir, exist_ok=True)
                for j in range(batch_size):
                    im = img[j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1,2,0))
                    im = Image.fromarray(im)
                    fullpath = os.path.join(opts.save_dir, f"{batch['image_id'][j]}.png")
                    im.save(fullpath)

        image_features = torch.cat(image_features)[:r]
        text_features = torch.cat(text_features)[:r]
        image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = torch.mm(image_norm, text_norm.T)
        _, inds = torch.topk(scores, k=R, largest=True)
        target = torch.arange(0, 100, device=opts.device)
        correct.append(torch.mean(inds.T.eq(target).float()))
        try:
            cor = inds.T.eq(target)
            keys = np.array([y for x in keys for y in x][:r])
            print(keys[cor[0].cpu()], len(keys[cor[0].cpu()]))
        except:
            print("need to debug")

    results["r_prec"] = torch.mean(torch.tensor(correct))
    return results["r_prec"]

# def compute_clip_r_precision(opts, num_gen=30000, R=1, r=100):
#     assert num_gen % r == 0
#     import clip
#     import torchvision.transforms as transforms
#     import numpy as np
#     model = opts.clip
#     data_iter = iter(opts.data_loader)
#     correct = []
#     results = dict()

#     for _ in tqdm(range(num_gen // r)):
#         r_count = 0
#         image_features = []
#         text_features = []
#         # 1: gt, 99: mismatch 
#         while r_count < r:
#             images = []
#             batch = next(data_iter)
#             batch_size = opts.batch_size #batch["image"].size(0)
#             for key in batch:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(opts.device)
#             tokens, tok_lens = batch["damsm_tokens"], batch["damsm_lengths"]
#             hidden = opts.text_encoder.init_hidden(tokens.size(0))
#             _, sent_embs = opts.text_encoder(tokens, tok_lens, hidden)
#             z = torch.randn([batch_size, opts.G.noise_size], device=opts.device)
#             #img = opts.G(z, sent_embs)
#             img = batch["image"]
#             img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             images.append(img)
#             images = torch.cat(images)
#             texts = clip.tokenize(batch["caption"]).to(opts.device)
#             # clip preprocess
#             images = torch.nn.functional.interpolate(images, (224, 224))
#             images = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(images.float())
#             img_feat = model.encode_image(images)
#             text_feat = model.encode_text(texts)
#             image_features.append(img_feat)
#             text_features.append(text_feat)
#             r_count += batch_size
#         image_features = torch.cat(image_features)[:r]
#         text_features = torch.cat(text_features)[:r]
#         image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
#         scores = torch.mm(image_features, text_features.T)
#         _, inds = torch.topk(scores, k=R, largest=True)
#         target = torch.arange(0, 100, device=opts.device)
#         correct.append(torch.mean(inds.T.eq(target).float()))
#     results["r_prec"] = torch.mean(torch.tensor(correct))
#     return results["r_prec"]


def compute_pr(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    detector_kwargs = dict(return_features=True)

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all_torch().to(torch.float16).to(opts.device)

    gen_features = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_torch().to(torch.float16).to(opts.device)

    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if opts.rank == 0 else None)
        kth = torch.cat(kth) if opts.rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else 'nan')
    return results['precision'], results['recall']

#----------------------------------------------------------------------------
