import sys
import os
import pathlib

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from train import train
from mona.config import config


import datetime

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()

    # elif sys.argv[1] == 'sample':
    #     folder = pathlib.Path("samples")
    #     if not folder.is_dir():
    #         os.mkdir(folder)
    #
    #     for i in range(100):
    #         im, img_processed = generate_image_sample()
    #         im.save(f"samples/{i}_raw.png")
    #         img_processed.save((f"samples/{i}_p.png"))
