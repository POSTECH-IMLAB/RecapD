import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List
from nltk.tokenize import RegexpTokenizer
import cv2
from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
from itertools import chain, repeat

class CocoCaptionsDataset(Dataset):
    r"""
    A PyTorch dataset to read COCO Captions dataset and provide it completely
    unprocessed. This dataset is used by various task-specific datasets
    in :mod:`~core.data.datasets` module.

    Args:
        data_root: Path to the COCO dataset root directory.
        split: Name of COCO 2014 split to read. One of ``{"train", "test"}``.
    """

    def __init__(self, data_root: str, split: str, use_damsm_emb: bool = False):

        # Get paths to image directory and annotation file.
        self.use_damsm_emb = use_damsm_emb
        image_dir = os.path.join(data_root, f"images")
        # for matching damsm
        filenames = pickle.load(open(os.path.join(data_root,f"{split}/filenames.pickle"),'rb'))
        x = pickle.load(open(os.path.join(data_root, 'captions.pickle'),'rb'))
        self.i2w = x[2]
        self.w2i = x[3]
        tokens = x[0] if split == "train" else x[1]
        del x

        captions, damsm_tokens = self.load_text_data(tokens)
        if self.use_damsm_emb:
            self.damsm_tokens = damsm_tokens

        self.instances = [
            (name, os.path.join(image_dir, f"{name}.jpg"), captions[i])
            for i,name in enumerate(filenames)
        ]

    def load_text_data(self, tokens):
        captions = []
        damsm_tokens = []
        for tok in tokens:
            cap = []
            damsm = []
            while len(cap) < 5:
                cap.append(' '.join([self.i2w[i] for i in tok])+'.')
                damsm.append(tok)
            captions.append(cap)
            damsm_tokens.append(damsm)
        return captions, damsm_tokens


        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_id, image_path, captions = self.instances[idx]
        choice_idx = np.random.choice(5)
        caption = captions[choice_idx]

        # shape: (height, width, channels), dtype: uint8
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_damsm_emb:
            damsm_tokens = self.damsm_tokens[idx][choice_idx]
            return {"image_id": image_id, "image": image, "caption": caption, "damsm_tokens":damsm_tokens}
        else:
            return {"image_id": image_id, "image": image, "caption": caption, }
