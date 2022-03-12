# Online validation w/ lots of samples w/o data generation
# 一边生成验证数据一边验证，可以进行超大规模验证而不需要担心内存/磁盘不够。

'''
这个工具可以同时online validate多个模型，并且更高效，因为不需要为每个模型单独生成新的data。
这个工具会创建多个pytorch进程，每个进程对应一个模型。
生成的data会同时喂给多个模型进行验证。

注：在线生成data其实是挺耗费资源的，因为要渲染文字。
'''

# FIXME: program does not end gracefully
# FIXME: how to log to file? stdout redicrect would deadlock
# TODO: reduce VRAM footprint per process, can't do so with PyTorch

from numpy.core.shape_base import block
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.multiprocessing as Mp

from mona.text import index_to_word
from mona.nn.model import Model
from mona.datagen.datagen import generate_image
from mona.config import config

import numpy as np
from PIL import Image

import sys
import time
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"

num_samples = 40000000
batch_size = 128
max_plot_incorrect_sample = 0
print_per_batch = 1000

class MyOnlineDataSet(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        im, text = generate_image()
        tensor = transforms.ToTensor()(im)
        return tensor, text


def par_validate_worker(model_file_path: str, input_queue: Mp.Queue):

    print('Worker Loading', model_file_path)
    net = Model(len(index_to_word)).to(device)
    net.load_state_dict(torch.load(
        model_file_path, map_location=torch.device(device)))
    net.eval()

    err = 0
    total = 0
    last_time = time.time()
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # print(model_file_path, 'h1', sample_idx)
            x, label = input_queue.get(block=True)

            # print(sample_idx, 'h2', label)
            # GPU code
            x = x.to(device)

            # GPU inference then CPU post-processing
            # This net.predict() can be slow because of GPU-CPU communication.
            predict = net.predict(x)
            for i in range(len(label)):
                pred = predict[i]
                truth = label[i]

                if pred != truth:
                    err += 1
                    err_out = f"== net: {model_file_path} pred: {pred}, truth: {truth} =="
                    print(err_out)
                    # TODO append to out queue

            # Stats
            # err += sum([0 if predict[i] == label[i]
            #             else 1 for i in range(len(label))])
            total += len(label)
            tput = int(total / (time.time() - last_time))
            # print('h3', sample_idx)

            if (total / batch_size) % print_per_batch == 0:
                print(str.format("{} tput {}, err rate {:.2e}. Tested {}, err {}",
                                 model_file_path, tput, err / total, total, err))

    # Finished
    print(str.format("DONE. {} tput {}, err rate {:.2e}. Tested {}, err {}",
          model_file_path, tput, err / total, total, err))
    return err


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Validate a model using online generated data from datagen')

    parser.add_argument('model_file', type=str, nargs='+',
                        help='A list of model file. e.g. m1.pt m2.pt')

    args = parser.parse_args()
    model_file_path_list = args.model_file
    print(f"Validating {len(model_file_path_list)} models: {model_file_path_list}")

    Mp.set_start_method('spawn')
    pool = Mp.Pool(len(model_file_path_list))
    manager = Mp.Manager()
    queue_list = []
    workers = []
    # Load a list of models to validate
    for model_file_path in model_file_path_list:
        print('ss', model_file_path)
        queue = Mp.Queue(maxsize=50)
        queue_list.append(queue)
        worker = Mp.Process(target=par_validate_worker, args=(model_file_path, queue))
        worker.start()
        # worker = pool.apply_async(par_validate_worker, model_file_path, queue)

    # Start workers
    # pool.starmap_async(par_validate_worker, zip(model_file_path_list, queue_list))
    print('workers started')

    validate_dataset = MyOnlineDataSet(num_samples)
    validate_loader = DataLoader(
        validate_dataset, batch_size=batch_size, num_workers=config["dataloader_workers"])

    for x, label in validate_loader:
        # for queue in queue_list:
        # print(label)
        for queue in queue_list:            
            queue.put((x, label), block=True)

    for worker in workers:
        worker.join()
    
    print('Closing threadpool')
    pool.close()

