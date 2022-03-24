import torch, torchvision
import torch.nn as nn
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    m = Model()
    print(m)

    optim = torch.optim.SGD(m.parameters(), lr=0.01)
    loss = CrossEntropyLoss()
    losslist=[]
    # 一轮学习
    # for data in dataloader:
    #     imgs, target = data
    #     outputs = m(imgs)
    #     result_loss = loss(outputs, target)
    #     optim.zero_grad()
    #     result_loss.backward()
    #     optim.step()
    #     losslist.append(result_loss.item())
    #     print(result_loss.item())
    # plt.plot(range(len(losslist)),losslist)
    # plt.show()
    for i in range(5):
        running_loss = 0.0
        for data in dataloader:
            imgs, target = data
            optim.zero_grad()
            outputs = m(imgs)
            result_loss = loss(outputs, target)
            optim.zero_grad()
            result_loss.backward()
            optim.step()
            running_loss = running_loss + result_loss
        print(running_loss)
