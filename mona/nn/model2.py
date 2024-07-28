import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from mona.nn.mobile_net_v3 import MobileNetV3Small
from mona.nn.svtr import MixingBlock, SVTRNet, PositionalEncoding


class Model2(nn.Module):
    def __init__(self, lexicon_size, in_channels, depth=2, hidden_channels=120, num_heads=8):
        super(Model2, self).__init__()
        self.cnn = MobileNetV3Small(out_size=hidden_channels, in_channels=in_channels)
        self.pe = PositionalEncoding(dim=hidden_channels, length=24)

        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.blocks = nn.Sequential()
        for i in range(depth):
            block = MixingBlock(
                dim=hidden_channels,
                num_heads=num_heads,
                is_local=False,
                drop_path=0.0,
                hw=None,
                input_length=24,
                mlp_ratio=2,
                attention_drop=0.1,
                drop=0.1,
            )
            self.blocks.add_module(f"mix{i}", block)
        self.norm = nn.LayerNorm(hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, lexicon_size)

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        x = x.squeeze(2).permute((0, 2, 1))
        x = self.pe(x)
        x = self.linear1(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.linear2(x)
        x = x.permute((1, 0, 2))

        x = F.log_softmax(x, dim=2)
        return x
    # def load_can_load(self, pretrained_dict, old_idx2word_path="models/index_2_word.json"):
    #     try:
    #         self.load_state_dict(pretrained_dict, strict=False)
    #     except Exception as e:
    #         print("[warning] just load can load")
    #         model_dict = self.state_dict()
    #         for k, v in pretrained_dict.items():
    #             if k in model_dict and v.size() == model_dict[k].size():
    #                 model_dict[k] = v
    #             elif k in model_dict and v.size() != model_dict[k].size():
    #                 print(f"size dismatch: {k}, {v.size()} -> {model_dict[k].size()}")
    #
    #                 # using word map to fill the known part of the linear2
    #                 if k != "linear2.weight" and k != "linear2.bias":
    #                     print(f"fail to load {k}")
    #                 old_idx2word = json.load(open(old_idx2word_path, "r"))
    #                 # make the key from str -> num
    #                 old_idx2word = {int(k): v for k, v in old_idx2word.items()}
    #                 old_word2idx = {}
    #                 for idx, word in old_idx2word.items():
    #                     old_word2idx[word] = idx
    #                 new_idx2word = index_to_word
    #                 new_word2idx = word_to_index
    #
    #                 # just check the new word
    #                 for i in range(model_dict[k].size()[0]):
    #                     if new_idx2word[i] not in old_word2idx:
    #                         print(f"{i} new word: {new_idx2word[i]}")
    #
    #                 # for linear2.x's matrix
    #                 # from [xx, ] -> [xx+m, ], where m is the number of new words
    #                 for i in range(v.size()[0]):
    #                     if old_idx2word[i] in new_word2idx:
    #                         new_i = new_word2idx[old_idx2word[i]]
    #                         model_dict[k][new_i] = v[i]
    #                         # print(f"load {old_idx2word[i]} -> {new_idx2word[new_i]}")
    #                     else:
    #                         print(f"fail to load {old_idx2word[i]}")
    #         self.load_state_dict(model_dict)
    # def freeze_backbone(self):
    #     for param in self.cnn.parameters():
    #         param.requires_grad = False
    # def unfreeze_backbone(self):
    #     for param in self.cnn.parameters():
    #         param.requires_grad = True
