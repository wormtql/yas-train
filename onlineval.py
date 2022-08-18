# Online validation w/ lots of samples w/o data generation
# 一边生成验证数据一边验证，可以进行超大规模验证而不需要担心内存/磁盘不够。

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from mona.text import index_to_word
from mona.nn.model import Model
from mona.nn.model2 import Model2
from mona.datagen.datagen import generate_image
from mona.config import config
from mona.nn import predict as predict_net

import numpy as np
from PIL import Image

import sys
import time
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"


class MyOnlineDataSet(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        im, text = generate_image()
        tensor = transforms.ToTensor()(im)
        return tensor, text


if __name__ == "__main__":
    # crnn
    # net = Model(len(index_to_word)).to(device)
    # svtr
    net = Model2(len(index_to_word), 1).to(device)
    # net = Model2(len(index_to_word), 1, hidden_channels=128, num_heads=4).to(device)

    parser = argparse.ArgumentParser(
        description='Validate a model using online generated data from datagen')
    parser.add_argument('model_file', type=str,
                        help='The model file. e.g. model_training.pt')

    args = parser.parse_args()
    model_file_path = args.model_file

    print(f"Validating {model_file_path}")
    net.load_state_dict(torch.load(
        model_file_path, map_location=torch.device(device)))

    batch_size = 32
    max_plot_incorrect_sample = 10
    num_samples = 1000000

    validate_dataset = MyOnlineDataSet(num_samples)
    validate_loader = DataLoader(
        validate_dataset, batch_size=batch_size, num_workers=config["dataloader_workers"])

    net.eval()
    err = 0
    total = 0
    last_time = time.time()
    with torch.no_grad():
        for x, label in validate_loader:
            x = x.to(device)
            # print(label)
            predict = predict_net(net, x)
            for i in range(len(label)):
                pred = predict[i]
                truth = label[i]

                # if True:
                if pred != truth:
                    print(f"\033[2K\r==== pred: {pred}, truth: {truth} ====")
                    # Save the incorrect samples
                    if err < max_plot_incorrect_sample:
                        arr = x.to('cpu')[i].squeeze()
                        im = Image.fromarray(np.uint8(arr * 255))
                        # im.show()
                        im.save(f"samples/err-sample-id{total+i}.png")

            # Stats
            err += sum([0 if predict[i] == label[i]
                        else 1 for i in range(len(label))])
            total += len(label)
            tput = int(total / (time.time() - last_time))
            print(str.format("Tput {} sample/s, err rate {:.2e}. Tested {}, err {}",
                             tput, err / total, total, err), end='\r')

    print(f"\nValidation result: {model_file_path} total {total} err {err}")
