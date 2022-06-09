import torch
from mona.text import index_to_word


def arr_to_string(arr):
    temp = ""
    last_word = "-"
    for word in arr:
        if word != last_word and word != "-":
            temp += word
        last_word = word
    return temp

def predict(net, x):
    y = net(x)
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

        ans.append(arr_to_string(words))
    return ans
