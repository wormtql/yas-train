import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from .cnn import MobileNetV1
from .mobile_net_v2 import MobileNetV2
from .mobile_net_v3 import MobileNetV3Small

from mona.text import index_to_word


class Model(nn.Module):
    def __init__(self, lexicon_size):
        super(Model, self).__init__()

        hidden_size = 128

        # self.cnn = MobileNetV1(3)
        # self.cnn = MobileNetV2(in_channels=1)
        self.cnn = MobileNetV3Small(in_channels=1, out_size=512)
        # self.gru = nn.GRU(
        #     input_size=512,
        #     hidden_size=hidden_size,
        #     bidirectional=True,
        #     num_layers=2,
        #     # dropout=0.25
        # )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=2,
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
        y, indices = torch.max(y, dim=2)

        indices.transpose_(0, 1)
        batch_size = x.size(0)

        ans = []
        for i in range(batch_size):
            words = []
            for j in indices[i]:
                word = index_to_word[j.item()]
                words.append(word)

            ans.append(self.arr_to_string(words))
        return ans