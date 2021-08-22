import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from mona.text import index_to_word, word_to_index
from mona.nn.model import Model
from mona.datagen.datagen import generate_image


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
            correct += sum([1 if predict[i] == label[i] else 0 for i in range(len(label))])
            total += len(label)

    net.train()
    return correct / total


def train():
    net = Model(len(index_to_word)).to(device)

    train_x = torch.load("data/train_x.pt")
    train_y = torch.load("data/train_label.pt")
    validate_x = torch.load("data/validate_x.pt")
    validate_y = torch.load("data/validate_label.pt")

    train_dataset = MyDataSet(train_x, train_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    validate_dataset = MyDataSet(validate_x, validate_y)
    validate_loader = DataLoader(validate_dataset, batch_size=64)

    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer = optim.Adadelta(net.parameters())
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)

    done = False
    max_rate = 0
    max_epoch = 0
    for epoch in range(50):
        print("epoch:", epoch)
        batch = 0
        for x, label in train_loader:
            optimizer.zero_grad()
            target_vector, target_lengths = get_target(label)
            target_vector, target_lengths = target_vector.to(device), target_lengths.to(device)
            x = x.to(device)

            batch_size = x.size(0)

            y = net(x)

            input_lengths = torch.full((batch_size,), 28, device=device, dtype=torch.long)
            loss = ctc_loss(y, target_vector, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"#{batch}: loss: {loss.item()}, {batch * 64 / 50000 * 100}%")
            # if (batch + 1) % 100 == 0:
            #     rate = validate(net, validate_loader)
            #     print(f"rate: {rate * 100}%")
            #     if rate > 0.99:
            #         done = True
            #         break

            batch += 1

        rate = validate(net, validate_loader)
        if rate > max_rate:
            max_rate = rate
            max_epoch = epoch
        print(f"rate: {rate * 100}%")
        if rate > 0.999:
            done = True
            break

        # if (batch + 1) % 100 == 0:
        torch.save(net.state_dict(), f"models/model_epoch_{epoch}.pt")
        if done:
            break

    print(f"max rate: {max_rate}, epoch: {max_epoch}")
    # net.to("cpu")
    temp = []
    temp2 = []
    for _ in range(10):
        test_image, test_text = generate_image()
        test_image = transforms.ToTensor()(test_image)
        test_image.unsqueeze_(dim=0)
        temp.append(test_image)
        temp2.append(test_text)
        # test_image = test_image.to(device)
        # predicted = net.predict(test_image)
        # print(predicted, test_text)
    inp = torch.cat(temp, dim=0)
    inp = inp.to(device)
    pred = net.predict(inp)
    print(pred)
    print(temp2)

    for x, label in validate_loader:
        x = x.to(device)
        predict = net.predict(x)
        print("predict:     ", predict[:10])
        print("ground truth:", label[:10])
        break


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


