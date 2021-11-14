# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
from . import dnnlib

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, encoder=None, preprocess=None, text_encoder=None,
    num_gpus=1, rank=0, device=None, data_loader = None, batch_size = 32, cache=True, **kwargs):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.encoder        = encoder  
        self.preprocess     = preprocess
        self.text_encoder   = text_encoder
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.cache          = cache
        self.data_loader    = data_loader
        self.batch_size     = batch_size

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=True, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------
def compute_damsm_feature_stats_for_dataset(opts, max_items=None, **stats_kwargs):

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'clip-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(opts.data_loader) * opts.batch_size
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    encoder = opts.encoder

    # Main loop.
    for images, _labels in opts.data_loader:
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = encoder(images.to(opts.device))
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats


def compute_clip_feature_stats_for_dataset(opts, max_items=None, **stats_kwargs):

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'clip-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(opts.data_loader) * opts.batch_size
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    clip = opts.clip

    # Main loop.
    for images, _labels in opts.data_loader:
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = clip.encode_image(images.to(opts.device))
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------
def compute_damsm_feature_stats_for_generator(opts,  **stats_kwargs):
    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    encoder = opts.encoder
    data_iter = iter(opts.data_loader)

    # Main loop.
    while not stats.is_full():
        images = []
        batch = next(data_iter)
        batch_size = batch["image"].size(0)
        for _i in range(batch_size):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(opts.device)
            tokens, tok_lens = batch["damsm_tokens"], batch["damsm_lengths"]
            hidden = opts.text_encoder.init_hidden(tokens.size(0))
            _, sent_embs = opts.text_encoder(tokens, tok_lens, hidden)
            z = torch.randn([batch_size, opts.G.z_dim], device=opts.device)
            img = opts.G(z, sent_embs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = encoder(images)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
    return stats


def compute_clip_feature_stats_for_generator(opts,  **stats_kwargs):
    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    clip = opts.clip 
    data_iter = iter(opts.data_loader)

    # Main loop.
    while not stats.is_full():
        images = []
        batch = next(data_iter)
        batch_size = batch["image"].size(0)
        for _i in range(batch_size):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(opts.device)
            tokens, tok_lens = batch["damsm_tokens"], batch["damsm_lengths"]
            hidden = opts.text_encoder.init_hidden(tokens.size(0))
            _, sent_embs = opts.text_encoder(tokens, tok_lens, hidden)
            z = torch.randn([batch_size, opts.G.z_dim], device=opts.device)
            img = opts.G(z, sent_embs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = clip.encode_image(images)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
    return stats

#----------------------------------------------------------------------------