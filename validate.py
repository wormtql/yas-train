import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from mona.text import index_to_word, word_to_index
from mona.nn.model import Model
from mona.datagen.datagen import generate_image
from mona.config import config


device = "cuda" if torch.cuda.is_available() else "cpu"


# a list of target strings
def get_target(s):
    target_length = []

    target_size = 0
    for i, target in enumerate(s):
        target_length.append(len(target))
        target_size += len(target)

    target_vector = []
    for target in s:
        for char in target:
            index = word_to_index[char]
            if index == 0:
                print("error")
            target_vector.append(index)

    target_vector = torch.LongTensor(target_vector)
    target_length = torch.LongTensor(target_length)

    return target_vector, target_length


def validate(net, validate_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in validate_loader:
            x = x.to(device)
            predict = net.predict(x)
            # print(predict)
            for i in range(len(label)):
                pred = predict[i]
                truth = label[i]
                if pred != truth:
                    print("pred:", pred, "truth:", truth)
            correct += sum([1 if predict[i] == label[i] else 0 for i in range(len(label))])
            total += len(label)

    return correct / total


def main():
    net = Model(len(index_to_word)).to(device)
    net.load_state_dict(torch.load(f"models/model_training.pt"))

    validate_x = torch.load("data/validate_x.pt")
    validate_y = torch.load("data/validate_label.pt")

    validate_dataset = MyDataSet(validate_x, validate_y)
    validate_loader = DataLoader(validate_dataset, batch_size=config["batch_size"])

    rate = validate(net, validate_loader)
    print(rate)


class MyDataSet(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.x[index]
        label = self.labels[index]

        return x, label


main()