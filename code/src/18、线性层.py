import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(64*3*32*32, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=64)
    m = Model()
    print(m)

    # for i, data in enumerate(dataloader):
    #     imgs, targets = data#torch.Size([64, 3, 32, 32])
    #     #imgs=torch.reshape(imgs,(1,1,1,-1))
    #     imgs=torch.flatten(imgs)#torch.Size([196608])
    #     output=m(imgs)#torch.Size([10])
    #     print(output.shape)
    #     break
