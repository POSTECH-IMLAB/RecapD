import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.abspath(__file__).split("scripts")[0])
from loguru import logger
import torch
from torch.utils.data import DataLoader

from capD.config import Config
from capD.data import ImageDirectoryDataset
from capD.factories import DiscriminatorFactory, TokenizerFactory, PretrainingModelFactory
from capD.utils.checkpointing import CheckpointManager
from capD.utils.common import common_parser
from capD.utils.metrics import CocoCaptionsEvaluator
from tqdm import tqdm


# fmt: off
parser = common_parser(
    description="""Run image captioning inference on a pretrained model, and/or
    evaluate pretrained model on COCO Captions val2017 split."""
)
parser.add_argument(
    "--data-root", default="/work/val2014",
    help="""Path to a directory containing image files to generate captions for.
    Default: COCO val2017 image directory as expected relative to project root."""
)
parser.add_argument(
    "--output", default="fake.json",
    help="Path to save predictions as a JSON file."
)
parser.add_argument(
    "--calc-metrics", action="store_true",
    help="""Calculate CIDEr and SPICE metrics using ground truth COCO Captions.
    This flag should not be set when running inference on arbitrary images."""
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device (this will be zero here by default).
        device = torch.cuda.current_device()

    #_C = Config(_A.config, _A.config_override)
    _C = Config("configs/bicaptioning_R_50_L1_H2048.yaml")
    tokenizer = TokenizerFactory.from_config(_C)

    if _A.data_root is None:
        _A.data_root = os.path.join(_C.DATA.ROOT, "val2014")

    val_dataloader = DataLoader(
        ImageDirectoryDataset(_A.data_root),
        batch_size=24,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    # Initialize model from a checkpoint.
    #netD = DiscriminatorFactory.from_config(_C).to(device)
    #ITERATION = CheckpointManager(netD=netD).load("exps/df256_capGD/checkpoint_413880.pth")
    #netD.eval()
    netD = PretrainingModelFactory.from_config(_C) 
    ITERATION = CheckpointManager(model=netD).load("bicaptioning_R_50_L1_H2048.pth")
    netD.to(device)
    netD.eval()
    netD.requires_grad_(False)


    captions = json.load(
        open(os.path.join("datasets/coco", "annotations", f"captions_val2014.json"))
    )

    path2id: Dict[int, str] = {
        im["file_name"].replace(".jpg", ""): im["id"]
        for im in captions["images"]
    }

    # Make a list of predictions to evaluate.
    predictions: List[Dict[str, Any]] = []

    for val_batch in tqdm(val_dataloader):

        val_batch["image"] = val_batch["image"].to(device)
        with torch.no_grad():
            output_dict = netD(val_batch["image"])

        # Make a dictionary of predictions in COCO format.
        for image_id, caption in zip(
            val_batch["image_id"], output_dict["predictions"]
        ):
            predictions.append(
                {
                    # Convert image id to int if possible (mainly for COCO eval).
                    "image_id": path2id[image_id[-25:]],
                    "caption": tokenizer.decode(caption.tolist()),
                }
            )

    logger.info("Displaying first 25 caption predictions:")
    for pred in predictions[:25]:
        logger.info(f"{pred['image_id']} :: {pred['caption']}")

    # Save predictions as a JSON file if specified.
    if _A.output is not None:
        #os.makedirs(os.path.dirname(_A.output), exist_ok=True)
        json.dump(predictions, open(_A.output, "w"))
        logger.info(f"Saved predictions to {_A.output}")

    # Calculate CIDEr and SPICE metrics using ground truth COCO Captions. This
    # should be skipped when running inference on arbitrary images.
    if True or _A.calc_metrics:
        # Assume ground truth (COCO val2017 annotations) exist.
        gt = os.path.join(_C.DATA.ROOT, "annotations", "captions_val2014.json")

        metrics = CocoCaptionsEvaluator(gt).evaluate(predictions)
        logger.info(f"Iter: {ITERATION} | Metrics: {metrics}")


if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus_per_machine > 1:
        raise ValueError("Using multiple GPUs is not supported for this script.")

    # No distributed training here, just a single process.
    main(_A)
