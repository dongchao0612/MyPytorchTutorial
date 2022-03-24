import torch, torchvision
import torch.nn as nn
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmod1 = nn.Sigmoid()

    def forward(self, input):
        #output=self.relu1(input)
        output = self.sigmod1(input)
        return output


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=64,shuffle=True, num_workers=0, drop_last=True)
    m = Model()
    print(m)
    # input_data = torch.tensor(
    #     [[1, -0.5],
    #      [-1, 3]]
    # )
    # input_data = torch.reshape(input_data, (-1, 1, 2, 2))
    # output_data = m(input_data)
    # print(output_data)
    writer = SummaryWriter("runs")
    for i, data in enumerate(dataloader):
        imgs, targets = data
        output = m(imgs)  # torch.Size([64, 3, 32, 32])
        #writer.add_images("input", imgs, i)
        writer.add_images("output_sigmod", output, i)
    writer.close()