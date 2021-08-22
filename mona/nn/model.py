import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from .cnn import MobileNetV1

from mona.text import index_to_word


class Model(nn.Module):
    def __init__(self, lexicon_size):
        super(Model, self).__init__()

        hidden_size = 128

        self.cnn = MobileNetV1(3)
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=2,
            # dropout=0.2
        )
        self.embedding = nn.Linear(hidden_size * 2, lexicon_size)
        self.softmax = nn.LogSoftmax(2)

    def forward(self, x):
        x = self.cnn(x)
        # convert CNN output to rnn input
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        y, _ = self.gru(x)
        y = self.embedding(y)
        y = F.log_softmax(y, dim=2)

        return y

    # def backward_hook(self, module, grad_input, grad_output):
    #     for g in grad_input:
    #         g[g != g] = 0

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