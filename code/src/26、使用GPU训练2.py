import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# 搭建神经网络
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mymodule = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.mymodule(x)
        return x


if __name__ == '__main__':
    # 准备训练数据
    train_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                                              download=True)
    # 准备测试数据
    test_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                             download=True)
    # 获取数据集长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"训练数据集的长度：{train_data_size}，测试数据集的长度：{test_data_size}")  # 50000，10000
    # 利用DataLoader加载数据集
    train_data_loader = DataLoader(train_data, batch_size=64)
    test_data_loader = DataLoader(test_data, batch_size=64)
    print(f"训练数据loader的长度：{train_data_loader.__len__()}，测试数据loader的长度：{test_data_loader.__len__()}")  # ：782，157

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # # 创建神经网络
    m = Model()
    m.to(device)
    # input_data=torch.randn((64,3,32,32))
    # output_data=m(input_data)#torch.Size([64, 10])

    # # 创建损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    # #添加tensorboard
    writer = SummaryWriter("runs")
    # 定义优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate)
    # 设置训练网络的一些参数
    total_train_step = 0  # 记录训练次数
    total_test_step = 0  # 记录测试次数
    epoch = 100  # 记录训练轮数
    for i in range(epoch):
        print(f"---------第 {i + 1} 轮训练开始---------")
        m.train()
        total_train_loss = 0
        for data in train_data_loader:
            imgs, target = data
            imgs=imgs.to(device)
            target=target.to(device)
            output = m(imgs)
            loss = loss_fn(output, target)  # 损失值
            # 优化器优化模型
            optimizer.zero_grad()  # 优化器梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化
            total_train_loss = total_train_loss + loss.item()
            total_train_step += 1

            if total_train_step % 100 == 0:
                print(f"训练次数 {total_train_step} ， 测试损失值 {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        print(f"整体训练集的Loss：{total_train_loss}")
        writer.add_scalar("total_train_loss", total_train_loss, global_step=i)
        # 测试步骤开始
        m.eval()
        total_test_loss = 0
        total_accuracy=0
        with torch.no_grad():
            for data in test_data_loader:
                imgs, target = data
                imgs = imgs.to(device)
                target = target.to(device)
                outputs = m(imgs)
                loss = loss_fn(outputs, target)
                total_test_loss = total_test_loss + loss.item()
                accuacy = (outputs.argmax(1) == target).sum()
                total_accuracy +=  accuacy
                total_test_step += 1
                if total_test_step % 10 == 0:
                    print(f"测试次数 {total_test_step} ， 测试损失值 {loss.item()}")
                writer.add_scalar("test_loss", loss.item(), total_test_step)

        print(f"整体测试集的Loss：{total_test_loss}")
        writer.add_scalar("total_test_loss", total_test_loss, i)
        print(f"整体测试集的Accuacy：{total_accuracy / test_data_size}")
        writer.add_scalar("total_accuracy", total_accuracy / test_data_size, i)

        torch.save(m, f"model/mymodule_{i}.pth")
        print("模型已经保存")
    writer.close()
