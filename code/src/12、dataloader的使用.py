import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # 准备测试数据集合
    test_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True,transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

    # 测试数据集第一张图片及其target
    # img, target = test_data[0]
    # print(img.shape, target,test_data.classes[target])#torch.Size([3, 32, 32]) 6 frog

    writer = SummaryWriter("runs")

    for i,data in enumerate(test_loader):
        imgs, target = data
        #print(imgs.shape,target) # 每次batch_size个数据
        writer.add_images("CIFAR10", imgs, i)
    writer.close()
