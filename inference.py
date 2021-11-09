import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image

from mona.nn.model import Model
from mona.text import word_to_index
from mona.datagen.datagen import generate_image
from mona.datagen.pre_process import pre_process


net = Model(len(word_to_index))

name = "model_training.pt"
net.load_state_dict(torch.load(f"models/{name}"))
net.eval()


def predict(image_name):
    im = Image.open(f"data/test/{image_name}")
    im = pre_process(im)
    im.save("test.png")

    tensor = T.ToTensor()(im)
    tensor.unsqueeze_(0)
    pred = net.predict(tensor)
    return pred[0]


names = [
    "1.jpg", "2.png", "3.png", "4.png", "5.png",
    "6.png", "7.png", "8.png", "9.png", "10.png",
    "11.png", "12.png", "13.png", "14.jpg", "15.png",
    "16.jpg", "17.png", "18.png", "19.png", "20.png",
    "21.png", "22.png", "23.png", "24.png", "25.png",
    "sample_0.png",
]
# names = ["25.png"]

for name in names:
    result = predict(name)
    print(f"{name}: {result}")

# wrong = 0
# wrong_list = []
# for i in range(64):
#     image, label = generate_image()
#     image = T.ToTensor()(image)
#     image.unsqueeze_(dim=0)
#     pred = net.predict(image)
#     if pred[0] != label:
#         wrong += 1
#         wrong_list.append((pred[0], label))
#     print(pred[0], label)
#
# print("wrong:", wrong)
# for pred, label in wrong_list:
#     print(pred, label)
