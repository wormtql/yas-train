import torch.nn as nn
import torch
import math


def drop_path(x, drop_prob = 0.0, training = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 is_local=False,
                 hw=(1, 1),
                 input_length=1,
                 attention_drop=0.1,
                 proj_drop=0.1,
                 local_kernel=(7, 11)
                 ):
        super(Attention, self).__init__()

        if hw is not None:
            self.hw = hw
            self.h = hw[0]
            self.w = hw[1]
            self.input_length = self.h * self.w
        else:
            self.input_length = input_length
        self.vector_length = dim
        self.is_local = is_local
        self.dim = dim

        self.num_heads = num_heads
        self.head_dimension = dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dimension)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_drop = nn.Dropout(attention_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # local attention is not used
        # if is_local:
        #     hk = local_kernel[0]
        #     wk = local_kernel[1]
        #     mask = torch.ones((self.input_length, self.h + hk - 1, self.w + wk - 1))
        #     for i in range(0, self.h):
        #         for j in range(0, self.w):
        #             mask[i * self.w + j, i:i + i * hk, j:j + wk] = 0.0
        #     mask_paddle = mask[:, hk // 2:self.h + hk // 2, wk // 2:self.w + wk // 2]
        #     mask_paddle = mask_paddle.flatten(1)
        #     mask_inf = torch.full((self.input_length, self.input_length), -1e9, dtype=torch.float32)
        #     # mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
        #     mask_paddle[mask_paddle < 0.5] = float("-inf")
        #     mask = mask_paddle.unsqueeze(0).unsqueeze(0)
        #     # self.mask = mask.unsqueeze(0).unsqueeze(0)
        #     self.register_buffer("mask", mask)

    def forward(self, x):
        # x: batch_size * input_length * vector_length
        qkv = self.qkv(x).reshape((-1, self.input_length, 3, self.num_heads, self.head_dimension))
        # qkv = self.qkv(x)
        qkv = qkv.permute((2, 0, 3, 1, 4))
        # now qkv is 3 * batch_size * head_index * input_length * head_dimension

        # q, v: batch_size * head_index * input_length * head_dimension
        q = qkv[0] * self.scale
        # k: batch_size * head_index * head_dimension * input_length
        k = qkv[1].permute((0, 1, 3, 2))
        v = qkv[2]

        # attention: batch_size * head_index * input_length * input_length
        attention = torch.matmul(q, k)
        # print(attention)
        if self.is_local:
            attention += self.mask
            # print(attention)
        attention = torch.softmax(attention, dim=-1)
        # print(attention)

        # x: batch_size * head_index * input_length * head_dimension
        x = torch.matmul(attention, v)
        # x: batch_size * input_length * head_index * head_dimension
        x = x.permute((0, 2, 1, 3))
        x = x.reshape((-1, self.input_length, self.dim))

        return x


class MLP(nn.Module):
    def __init__(self, in_features, ratio):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features * ratio)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features * ratio, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class MixingBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 is_local,
                 hw=(1, 1),
                 input_length=1,
                 drop_path=0.0,
                 drop=0.1,
                 attention_drop=0.1,
                 mlp_ratio=2,
                 ):
        super(MixingBlock, self).__init__()

        self.mixer = Attention(
            dim=dim,
            num_heads=num_heads,
            is_local=is_local,
            hw=hw,
            proj_drop=drop,
            attention_drop=attention_drop,
            input_length=input_length
        )

        # self.drop_path = DropPath(drop_path)
        # self.drop_path = nn.Identity()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm1 = nn.Identity()
        # self.norm2 = nn.Identity()

        self.mlp = MLP(in_features=dim, ratio=mlp_ratio)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        # x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self,
                 img_size=(32, 256),
                 in_channels=3,
                 embed_dim=64,
                 ):
        super(PatchEmbed, self).__init__()

        hidden_channel = embed_dim // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.GELU(),
            nn.Conv2d(hidden_channel, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.permute((0, 2, 1))
        return x


class SubSample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 1)):
        super(SubSample, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).permute((0, 2, 1))
        x = self.norm(x)
        return x


class Combining(nn.Module):
    def __init__(self, w, in_channels, out_channels):
        super(Combining, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, w))
        self.linear = nn.Conv2d(in_channels, out_channels, stride=(1, 1), kernel_size=(1, 1), padding=0)
        self.hs = nn.Hardswish()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(x)
        x = self.linear(x)
        x = self.hs(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim, length):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # print(pe.shape)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(x[0][0])
        # print(x.shape)
        x = x + self.pe
        # print(x.shape)
        # print(x[0][0])
        return x


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 act=nn.GELU):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SVTRNet(nn.Module):
    def __init__(self,
                 in_channels,
                 in_length,
                 out_channels,
                 hidden_channels,
                 depth=2,
                 num_heads=8):
        super(SVTRNet, self).__init__()

        # self.conv1 = ConvBNLayer(in_channels, in_channels // 8, padding=1, act=nn.SiLU)
        # self.conv2 = ConvBNLayer(in_channels // 8, hidden_dim, kernel_size=1, act=nn.SiLU)

        # self.conv1 = ConvBNLayer(in_channels, hidden_channels, padding=0, kernel_size=(1, 1), act=nn.SiLU)
        self.linear1 = nn.Linear(in_channels, hidden_channels)

        self.blocks = nn.Sequential()
        for i in range(depth):
            block = MixingBlock(
                dim=hidden_channels,
                num_heads=num_heads,
                is_local=False,
                drop_path=0.0,
                hw=None,
                input_length=in_length,
                mlp_ratio=2,
                attention_drop=0.1,
                drop=0.1,
            )
            self.blocks.add_module(f"mix{i}", block)

        self.norm = nn.LayerNorm(hidden_channels)
        # self.conv3 = ConvBNLayer(hidden_dim, in_channels, kernel_size=(1, 1), act=nn.SiLU)
        # self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, padding=1, act=nn.SiLU)
        # self.conv1x1 = ConvBNLayer(in_channels // 8, dim, kernel_size=(1, 1), act=nn.SiLU)
        self.linear2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        h = x
        print(x.shape)
        x = self.linear1(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.linear2(x)
        # x = x.permute((1, 0, 2))
        return x


# class SVTRNet(nn.Module):
#     def __init__(self,
#                  img_size=(32, 256),
#                  in_channels=3,
#                  embed_dim=(64, 128, 256),
#                  depth=(3, 6, 3),
#                  num_heads=(2, 4, 8),
#                  is_locals=(True,) * 6 + (False,) * 6,
#                  out_channels=192):
#         super(SVTRNet, self).__init__()
#
#         layer_count = sum(depth)
#         dpr = [i / (layer_count - 1) for i in range(layer_count)]
#
#         h = img_size[0] // 4
#         w = img_size[1] // 4
#         self.h = h
#         self.w = w
#
#         self.patch_embed = PatchEmbed(
#             img_size=img_size,
#             in_channels=in_channels,
#             embed_dim=embed_dim[0]
#         )
#         self.pe = PositionalEncoding(dim=embed_dim[0], length=h * w)
#
#         self.blocks1 = nn.Sequential()
#         for i in range(depth[0]):
#             block = MixingBlock(
#                 embed_dim[0],
#                 num_heads=num_heads[0],
#                 is_local=is_locals[0:depth[0]][i],
#                 hw=(h, w),
#                 drop_path=dpr[0:depth[0]][i]
#             )
#             self.blocks1.add_module(f"mix{i}", block)
#         self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], stride=(2, 2))
#
#         self.blocks2 = nn.Sequential()
#         for i in range(depth[1]):
#             block = MixingBlock(
#                 dim=embed_dim[1],
#                 num_heads=num_heads[1],
#                 is_local=is_locals[depth[0]:depth[0] + depth[1]][i],
#                 hw=(h // 2, w // 2),
#                 drop_path=dpr[depth[0]:depth[0] + depth[1]][i]
#             )
#             self.blocks2.add_module(f"mix{i}", block)
#         self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], stride=(2, 2))
#
#         self.blocks3 = nn.Sequential()
#         for i in range(depth[2]):
#             block = MixingBlock(
#                 dim=embed_dim[2],
#                 num_heads=num_heads[2],
#                 is_local=is_locals[depth[0] + depth[1]:][i],
#                 hw=(h // 4, w // 4),
#                 drop_path=dpr[depth[0] + depth[1]:][i]
#             )
#             self.blocks3.add_module(f"mix{i}", block)
#         self.combining = Combining(w // 4, embed_dim[2], out_channels)
#
#     def forward(self, x):
#         x = self.patch_embed(x)
#         # print(x.shape)
#         x = self.pe(x)
#         # print(x.shape)
#         x = self.blocks1(x)
#         # print(x.shape)
#         x = x.permute((0, 2, 1)).reshape((-1, x.shape[2], self.h, self.w))
#         # print(x.shape)
#         x = self.sub_sample1(x)
#         # print(x.shape)
#
#         x = self.blocks2(x)
#         x = x.permute((0, 2, 1)).reshape((-1, x.shape[2], self.h // 2, self.w // 2))
#         # print(x.shape)
#         x = self.sub_sample2(x)
#
#         x = self.blocks3(x)
#         # print(x)
#         x = x.permute((0, 2, 1)).reshape((-1, x.shape[2], self.h // 4, self.w // 4))
#         # print(x.shape)
#         x = self.combining(x)
#         x = x.squeeze(2)
#         x = x.permute((2, 0, 1))
#         x = torch.log_softmax(x, dim=2)
#
#         return x
