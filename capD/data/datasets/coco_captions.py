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
        split: Name of COCO 2017 split to read. One of ``{"train", "val"}``.
    """

    def __init__(self, data_root: str, split: str, use_damsm_emb: bool = False):

        # Get paths to image directory and annotation file.
        image_dir = os.path.join(data_root, f"images")
        captions = json.load(
            open(os.path.join(data_root, "annotations", f"captions_{split}2014.json"))
        )
        # Collect list of captions for each image.
        captions_per_image: Dict[int, List[str]] = defaultdict(list)
        for ann in captions["annotations"]:
            if 'agreeable' in ann["caption"]: # for matching damsm
                continue
            if len(captions_per_image[ann["image_id"]]) >= 5:
                continue
            captions_per_image[ann["image_id"]].append(ann["caption"])

        # Collect image file for each image (by its ID).
        image_filepaths: Dict[int, str] = {
            im["id"]: os.path.join(image_dir, im["file_name"])
            for im in captions["images"]
        }

        # Keep all annotations in memory. Make a list of tuples, each tuple
        # is ``(image_id, file_path, list[captions])``.
        instances = [
            (im_id, image_filepaths[im_id], captions_per_image[im_id])
            for im_id in captions_per_image.keys()
        ]

        self.instances = sorted(instances, key=lambda x: x[0])
        self.use_damsm_emb = use_damsm_emb
        if self.use_damsm_emb:
            x = pickle.load(open(os.path.join(data_root, 'captions.pickle'),'rb'))
            self.i2w = x[2]
            self.w2i = x[3]
            del x 

            damsm_path = os.path.join(data_root, f"{split}_damsm_tokens.pickle")
            if os.path.isfile(damsm_path):
                self.damsm_tokens = pickle.load(open(damsm_path, 'rb'))
            else:
                self.damsm_tokens = self.preprocess_damsm(damsm_path)
            #damsm = np.load(os.path.join(data_root, f'{split}_embeddings.npy'))
            #damsm = damsm.reshape(-1,5,256)
            # file = os.path.join(data_root, split, 'filenames.pickle')
            # filenames = pickle.load(open(file, 'rb'))
            # if split == 'train':
            #     captions = x[0] 
            # else:
            #     captions = x[1]
            # damsm_ids = np.argsort([name2id[os.path.join(image_dir,name)] for name in filenames])
            # damsm_ids = list(chain.from_iterable(repeat(id, 5) for id in damsm_ids))
            # self.damsm_tokens = [captions[i] for i in damsm_ids]

    def preprocess_damsm(self, damsm_path):
        damsm_tokens = [] 
        damsm_tokenizer = RegexpTokenizer(r'\w+')
        for im_id, file, captions in self.instances:
            caps = []
            for caption in captions:
                try:
                    caps.append([self.w2i[w] for w in damsm_tokenizer.tokenize(caption.lower())])
                except:
                    raise RuntimeError
            damsm_tokens.append(caps)

        with open(damsm_path, 'wb') as f:
            pickle.dump(damsm_tokens, f, protocol=2)

        print("Save damsm tokens")
        return damsm_tokens


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_id, image_path, captions = self.instances[idx]
        choice_idx = np.random.choice(len(captions))
        caption = captions[choice_idx]

        # shape: (height, width, channels), dtype: uint8
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_damsm_emb:
            damsm_tokens = self.damsm_tokens[idx][choice_idx]
            return {"image_id": image_id, "image": image, "caption": caption, "damsm_tokens":damsm_tokens}
        else:
            return {"image_id": image_id, "image": image, "caption": caption, }
