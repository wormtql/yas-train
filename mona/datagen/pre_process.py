import numpy as np
from PIL import Image

from mona.config import config


def to_gray(arr):
    arr = np.dot(arr, [0.2989, 0.5870, 0.1140])
    return arr


def normalize(arr, auto_inverse=True):
    arr -= np.min(arr)
    # arr -= arr[-1, -1]
    # arr = arr.clip(min=0)
    arr /= np.max(arr)
    # arr[arr < 0.6] = 0
    if auto_inverse and arr[-1, -1] > 0.5:
        arr = 1 - arr
    arr[arr < 0.6] = 0
    # if auto_inverse:
    #     arr -= arr[-1, -1]
    #     arr = arr.clip(min=0)
    #     arr /= np.max(arr)
    return arr


def my_func(arr):
    arr -= arr[0, 0]
    arr = arr.clip(min=0)
    return arr


def crop(arr, tol=0.7):
    mask = arr > tol
    m, n = arr.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return arr[row_start:row_end, col_start:col_end]


def resize_to_height(arr):
    im = Image.fromarray(np.uint8(arr * 255))
    w, h = im.size
    scale = config["height"] / h
    im = im.resize((int(w * scale), config["height"]), Image.BILINEAR)
    arr = np.array(im)
    return arr / 255


def pad_to_width(arr):
    if arr.shape[1] >= config["train_width"]:
        return arr[:, :config["train_width"]]
    return np.pad(
        arr, [[0, 0], [0, config["train_width"] - arr.shape[1]]], mode="constant", constant_values=0
    )


def to_numpy(im):
    arr = np.asarray(im, dtype=float)
    arr /= 255
    return arr


def pre_process(im):
    arr = to_numpy(im)
    arr = to_gray(arr)
    # arr = binarization(arr)
    # im = Image.fromarray(np.uint8(arr * 255))
    # im.show()
    arr = normalize(arr)
    # arr = my_func(arr)
    # im = Image.fromarray(np.uint8(arr * 255))
    # im.show()
    arr = crop(arr)
    arr = normalize(arr, False)
    # im = Image.fromarray(np.uint8(arr * 255))
    # im.show()
    arr = resize_to_height(arr)
    arr = pad_to_width(arr)

    im = Image.fromarray(np.uint8(arr * 255))
    return im