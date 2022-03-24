import torchvision
# 全局取消证书验证
import ssl
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    transforms = transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, download=True, transform=transforms)
    test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, download=True, transform=transforms)

    #print(test_set.classes)
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #img, target = train_set[0]
    #img.show()  # 未进行ToTensor
    # print(target)
    # print(test_set.classes[target])
    # print(train_set.data.shape)#(50000, 32, 32, 3)
    # print(test_set.data.shape)#(10000, 32, 32, 3)
    # print(test_set[0])#tensor

    writer = SummaryWriter("runs")
    for i in range(10):
        img, target = test_set[i]
        writer.add_image("CIFAR10", img, i)
    writer.close()
