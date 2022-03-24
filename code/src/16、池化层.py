import torch, torchvision
import torch.nn as nn
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, inpit):
        output = self.maxpool(inpit)
        return output


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=64,shuffle=True, num_workers=0, drop_last=True)
    m = Model()
    print(m)
    # input_data = torch.tensor(
    #     [[1, 2, 0, 3, 1],
    #      [0, 1, 2, 3, 1],
    #      [1, 2, 1, 0, 0],
    #      [5, 2, 3, 1, 1],
    #      [2, 1, 0, 1, 1]]
    #     , dtype=torch.float32)
    # input_data = torch.reshape(input_data, (-1, 1, 5, 5))
    # outout = m(input_data)
    # print(outout)
    writer = SummaryWriter("runs")
    for i, data in enumerate(dataloader):
        imgs, targets = data
        output = m(imgs)  # torch.Size([64, 3, 11, 11])
        writer.add_images("input", imgs, i)
        writer.add_images("output", output, i)
    writer.close()
