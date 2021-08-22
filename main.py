import sys
import os
import pathlib
# import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from mona.nn.cnn import Block, MobileNetV1

from mona.text.stat import random_stat, random_value
from mona.datagen.datagen import generate_image
from train import train

# parser = argparse.ArgumentParser(description="Genshin Impact Game Scanner")
# gen_parser = parser.add_subparsers(dest="gen")


if sys.argv[1] == "gen":
    train_size = 10000
    validate_size = 1000

    folder = pathlib.Path("data")
    if not folder.is_dir():
        os.mkdir(folder)

    x = []
    y = []
    for _ in range(train_size):
        im, text = generate_image()
        tensor = transforms.ToTensor()(im)
        tensor = torch.unsqueeze(tensor, dim=0)
        x.append(tensor)
        y.append(text)

    xx = torch.cat(x, dim=0)
    torch.save(xx, "data/train_x.pt")
    torch.save(y, "data/train_label.pt")

    x = []
    y = []
    for _ in range(validate_size):
        im, text = generate_image()
        tensor = transforms.ToTensor()(im)
        tensor = torch.unsqueeze(tensor, dim=0)
        x.append(tensor)
        y.append(text)

    xx = torch.cat(x, dim=0)
    torch.save(xx, "data/validate_x.pt")
    torch.save(y, "data/validate_label.pt")

    # generate sample
    for i in range(10):
        im, text = generate_image()
        im.save(f"data/sample_{i}.png")
elif sys.argv[1] == "train":
    train()
