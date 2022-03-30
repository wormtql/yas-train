from queue import PriorityQueue
import heapq

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from .cnn import MobileNetV1
from .mobile_net_v2 import MobileNetV2
from .mobile_net_v3 import MobileNetV3Small

from mona.text import index_to_word


def put_queue(q, item):
    q.put_nowait([item["score"], item])
    if q.full():
        q.get_nowait()


def decode_beam(y, beam_size=20):
    t = y.size(0)
    v = y.size(1)

    beam = PriorityQueue(maxsize=beam_size + 1)
    for i in range(v):
        entry = {
            "parent": None,
            "index": i,
            "score": y[0][i]
        }
        put_queue(beam, entry)
        # beam.put_nowait([-y[0][i], entry])
    # print(beam)

    for i in range(1, t):
        entries = []
        while not beam.empty():
            entries.append(beam.get_nowait()[1])

        for entry in entries:
            for j in range(v):
                new_score = entry["score"] + y[i, j]
                new_entry = {
                    "parent": entry,
                    "score": new_score,
                    "index": j
                }
                # beam.put_nowait([new_score, new_entry])
                put_queue(beam, new_entry)
        # print(beam)

    arr = []
    while beam.qsize() > 1:
        beam.get_nowait()
    e = beam.get_nowait()[1]
    while e is not None:
        arr.append(index_to_word[e["index"]])
        e = e["parent"]
    arr.reverse()

    # print(arr)
    return arr


class Model(nn.Module):
    def __init__(self, lexicon_size):
        super(Model, self).__init__()

        hidden_size = 128

        # self.cnn = MobileNetV1(3)
        # self.cnn = MobileNetV2(in_channels=1)
        self.cnn = MobileNetV3Small(in_channels=1, out_size=512)
        # self.rnn = nn.GRU(
        #     input_size=512,
        #     hidden_size=hidden_size,
        #     bidirectional=True,
        #     num_layers=2,
        #     dropout=0.2
        # )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
        )
        self.embedding = nn.Linear(hidden_size * 2, lexicon_size)
        self.softmax = nn.LogSoftmax(2)

    def forward(self, x):
        x = self.cnn(x)
        # convert CNN output to rnn input
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        y, _ = self.rnn(x)
        y = self.embedding(y)
        y = F.log_softmax(y, dim=2)

        return y

    def predict_pil(self, x):
        x = transforms.ToTensor()(x)
        x.unsqueeze_(dim=0)

        return self.predict(x)

    def arr_to_string(self, arr):
        temp = ""
        last_word = "-"
        for word in arr:
            if word != last_word and word != "-":
                temp += word
            last_word = word
        return temp


    def predict(self, x):
        y = self(x)
        # y = torch.transpose(y, 0, 1)
        batch_size = x.size(0)
        # print(y.size())

        # ans = []
        # for i in range(batch_size):
        #     arr = decode_beam(y[i], 2)
        #     ans.append(self.arr_to_string(arr))
        #     print(ans[-1])
        # print(ans)
        # return ans
        y, indices = torch.max(y, dim=2)

        indices.transpose_(0, 1)

        ans = []
        for i in range(batch_size):
            words = []
            for j in indices[i]:
                word = index_to_word[j.item()]
                words.append(word)

            ans.append(self.arr_to_string(words))
        return ans
