import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    m = Model()
    print(m)
    writer = SummaryWriter("runs")

    for i, data in enumerate(dataloader):
        imgs, targets = data
        output = m(imgs)  # torch.Size([64, 6, 30, 30])
        writer.add_images("input", imgs, i)
        output = torch.reshape(output, (-1, 3, 30, 30))  # torch.Size([128, 3, 30, 30])
        writer.add_images("output", output, i)
    writer.close()
