import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    #train_data=torchvision.datasets.ImageNet(root="ImageNet",split="train",download=True,transform=torchvision.transforms.ToTensor())
    vgg16_False = torchvision.models.vgg16(pretrained=False,progress=True)
    vgg16_True = torchvision.models.vgg16(pretrained=True,progress=True)

    #print(vgg16_True)
    train_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),download=True)
    #添加模型
    vgg16_True.classifier.add_module("add_linear", nn.Linear(1000, 10))
    #print(vgg16_True)
    #修改模型
    vgg16_False.classifier[6]=nn.Linear(4096,10)
    #print(vgg16_False)


    input_data = torch.ones((64, 3, 32, 32))
    output_data = vgg16_True(input_data)
    print(output_data.shape)  # torch.Size([64, 10])
    writer = SummaryWriter("runs")
    writer.add_graph(vgg16_True,input_data)
    writer.close()
