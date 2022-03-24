import torch, torchvision
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
    # input_data = torch.ones((64, 3, 32, 32))
    # output_data = m(input_data)
    # print(output_data.shape)  # torch.Size([64, 10])
    # writer = SummaryWriter("runs")
    # writer.add_graph(m,input_data)
    # writer.close()

    # 引入损失函数
    loss_cross = CrossEntropyLoss()
    for data in dataloader:
        imgs,target=data
        outputs=m(imgs)
        #print(outputs,target)
        #tensor([[ 0.0146, -0.1189,  0.1255, -0.0325, -0.1456,  0.1340, -0.1293, -0.0729,-0.0324, -0.0676]], grad_fn=<AddmmBackward0>)
        #tensor([3])
        result_cross = loss_cross(outputs, target)
        # result_cross.backward() 梯度更新
        print(result_cross)

