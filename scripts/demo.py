from copy import deepcopy
import os 
import sys



sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from typing import Any

import random
import numpy as np
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from scipy.stats import truncnorm
from PIL import Image

# fmt: off
from capD.config import Config
from capD.data.datasets.coco_captions import CocoCaptionsDataset
from capD.factories import (
   GeneratorFactory, TextEncoderFactory,
    
)
from capD.utils.checkpointing import CheckpointManager
from capD.utils.common import common_parser, common_setup

parser = common_parser(
    description="Train a capD model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default="exps/256_cond_cap/checkpoint.pth",
    help="Path to a checkpoint to resume training from (if provided)."
)
group.add_argument(
    "--manualSeed", default=10, type=int
)

group.add_argument(
    "--text", type=str,
)


def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
    return truncation * values

def gen_example(data_dic, _C, _A):
    # Build and load the generator
    text_encoder = TextEncoderFactory.from_config(_C).cuda()
    text_encoder.eval()
    netG_ema = GeneratorFactory.from_config(_C).cuda()
    iteration = CheckpointManager(netG_ema=netG_ema).load(_A.resume_from)
    netG_ema.eval()

    manualSeed = random.randint(1,100000)
    print(manualSeed)

    for key in data_dic:
        captions, cap_lens, sorted_indices = data_dic[key]

        batch_size = captions.shape[0]
        nz = 100
        captions = Variable(torch.from_numpy(captions), volatile=True)
        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        for i in range(1):  # 16
            #noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = torch.from_numpy(truncated_z_sample(batch_size, nz, seed=manualSeed)).float()
            #print(noise)
            noise = noise.cuda()
            #######################################################
            # (1) Extract text embeddings
            ######################################################
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            #######################################################
            # (2) Generate fake images
            ######################################################
            fake_imgs = netG_ema(noise, sent_emb)
            # G attention
            cap_lens_np = cap_lens.cpu().data.numpy()
            for j in range(batch_size):
                #save_name = '%d_s_%d' % (i, sorted_indices[j])
                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # print('im', im.shape)
                im = Image.fromarray(im)
                fullpath = '%s.png' % (key)
                im.save(fullpath)

def gen(wordtoix, _C, _A):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    #filepath = 'example_sentences.txt'
    data_dic = {}
        
    # a list of indices for a sentence
    captions = []
    cap_lens = []
    sent = _A.text.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    if len(tokens) == 0:
        print('sent', sent)
        return

    rev = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    key = sent 
    data_dic[key] = [cap_array, cap_lens, sorted_indices]
    gen_example(data_dic, _C, _A)



def main(_A: argparse.Namespace):
    device = torch.cuda.current_device()

    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    test_dataset = CocoCaptionsDataset(data_root = "datasets/coco", split="test")
    gen(test_dataset.w2i, _C, _A)

        
if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
