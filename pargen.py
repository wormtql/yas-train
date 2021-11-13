# Data generator, multi-threaded
# 多线程数据生成器，极大加快数据生成速度，并优化了内存占用。
# NOTE： 需要注意随机数生成的线程安全问题，目前没有遇到问题。
""" 
NOTE: Some of the random functions (e.g. random.gauss()) are not thread-safe, 
and would generate same values across different thread.

The functions used in this project (randint, randn) should be thread-safe.
"""

from typing import Tuple
from torch.functional import Tensor
from mona.config import config
import os
import pathlib
from multiprocessing import Pool

import torchvision.transforms as transforms
import torch
import datetime
from itertools import chain

from mona.datagen.datagen import generate_image


def progressBar(current, total, barLength=40):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def fill_data(target_tensor_slice: Tensor) -> list:
    """Fill a given tensor slice with generated image

    Args:
        target_tensor_slice (Tensor): Tensor slice to store the generated image

    Returns:
        list: List of labels
    """
    length = target_tensor_slice.shape[0]
    y = []
    for i in range(length):
        im, text = generate_image()
        tensor = transforms.ToTensor()(im)
        tensor = torch.unsqueeze(tensor, dim=0)
        # NOTE: here tensor.shape == [1, 1, 32, 384]
        target_tensor_slice[i] = tensor
        y.append(text)

        # Worker print progress
        if i % 100 == 0:
            progressBar(i, length)
    return y


def gen_dataset_with_label(size, threads=2) -> Tuple[Tensor, list]:
    # Allocate the output Tensor, and split into sub-tensors (views, copy-free)
    x = torch.zeros((size, 1, 32, 384))
    x_split = torch.tensor_split(x, threads, dim=0)

    with Pool(threads) as p:
        print(f"Starting threadpool with {threads} threads.")
        labels = p.map(fill_data, x_split)
        print("\nStopping threadpool.")
        return x, list(chain.from_iterable(labels))


if __name__ == '__main__':

    train_size = config["train_size"]
    validate_size = config["validate_size"]

    folder = pathlib.Path("data")
    if not folder.is_dir():
        os.mkdir(folder)

    # Use physical cores only
    threads = max(1, os.cpu_count() // 2)

    print(
        f"Train size {train_size}, Val size {validate_size}, Thread count {threads}")

    # Generate and save training set
    print(f"{datetime.datetime.now()} Generating training data")
    x, y = gen_dataset_with_label(size=train_size, threads=threads)

    print(f"{datetime.datetime.now()} Saving training data")
    torch.save(x, "data/train_x.pt")
    torch.save(y, "data/train_label.pt")

    print(f"{datetime.datetime.now()} Generating validation data")
    x, y = gen_dataset_with_label(size=validate_size, threads=threads)

    print(f"{datetime.datetime.now()} Saving validation data")
    torch.save(x, "data/validate_x.pt")
    torch.save(y, "data/validate_label.pt")

    # Verify the result
    # for tensor, y in zip(x,y):
    #     arr = tensor.squeeze()
    #     im = Image.fromarray(np.uint8(arr * 255))
    #     im.show()
    #     print(y)
    #     import time
    #     time.sleep(1)
