import sys
import os
import pathlib

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# from mona.text.stat import random_stat, random_value
from mona.datagen.datagen import generate_image, generate_image_sample
from train import train
from mona.config import config


import datetime

if __name__ == "__main__":
    # if sys.argv[1] == "gen":
        # train_size = config["train_size"]
        # validate_size = config["validate_size"]
        #
        # folder = pathlib.Path("data")
        # if not folder.is_dir():
        #     os.mkdir(folder)
        #
        # x = []
        # y = []
        # cnt = 0
        # for _ in range(train_size):
        #     im, text = generate_image()
        #     tensor = transforms.ToTensor()(im)
        #     tensor = torch.unsqueeze(tensor, dim=0)
        #     x.append(tensor)
        #     y.append(text)
        #
        #     cnt += 1
        #     if cnt % 1000 == 0:
        #         print(f'{cnt} / {train_size} {datetime.datetime.now()}')
        #
        # xx = torch.cat(x, dim=0)
        # torch.save(xx, "data/train_x.pt")
        # torch.save(y, "data/train_label.pt")
        #
        # x = []
        # y = []
        # for _ in range(validate_size):
        #     im, text = generate_image()
        #     tensor = transforms.ToTensor()(im)
        #     tensor = torch.unsqueeze(tensor, dim=0)
        #     x.append(tensor)
        #     y.append(text)
        #
        # xx = torch.cat(x, dim=0)
        # torch.save(xx, "data/validate_x.pt")
        # torch.save(y, "data/validate_label.pt")
        #
        # # generate sample
        # for i in range(50):
        #     im, text = generate_image()
        #     im.save(f"data/sample_{i}.png")
    if sys.argv[1] == "train":
        train()

    elif sys.argv[1] == 'sample':
        folder = pathlib.Path("samples")
        if not folder.is_dir():
            os.mkdir(folder)

        for i in range(100):
            im, img_processed = generate_image_sample()
            im.save(f"samples/{i}_raw.png")
            img_processed.save((f"samples/{i}_p.png"))
