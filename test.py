import torch

from mona.nn.mobile_net_v3 import MobileNetV3Small
from mona.nn.mobile_net_v2 import MobileNetV2

net = MobileNetV3Small(512)
# net = MobileNetV2()
net.eval()

x = torch.randn(3, 3, 32, 384)
y = net(x)
print(y.size())