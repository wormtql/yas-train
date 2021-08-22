import torch
import torchvision.transforms as T

from mona.nn.model import Model
from mona.text import word_to_index
from mona.datagen.datagen import generate_image


net = Model(len(word_to_index))

net.load_state_dict(torch.load("models/chs_all.pt"))
net.eval()


# temp = []
# labels = []
# for i in range(64):
#     image, label = generate_image()
#     image = T.ToTensor()(image)
#     image.unsqueeze_(dim=0)
#     temp.append(image)
#     labels.append(label)
#
# temp = torch.cat(temp, dim=0)
# pred = net.predict(temp)
# print("prediction:  ", pred)
# print("ground truth:", labels)

for i in range(64):
    image, label = generate_image()
    image = T.ToTensor()(image)
    image.unsqueeze_(dim=0)
    pred = net.predict(image)
    print(pred[0], label)
