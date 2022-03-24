import torch
import torchvision
from torch import nn

if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16(pretrained=False)
    # 模型保存方法1  模型结构 +模型参数
    # torch.save(vgg16, "model/vgg16_m1.pth")
    # 模型保存方法1 模型加载方法1
    # new_vgg16=torch.load("model/vgg16_m1.pth")
    # print(new_vgg16)

    # 模型保存方法2  模型参数(官方推荐)
    torch.save(vgg16.state_dict(),"model/vgg16_m2.pth")
    # 模型保存方法2 》模型加载方法2
    new_vgg16=torchvision.models.vgg16(pretrained=False)
    # new_vgg16 = torch.load("model/vgg16_m2.pth")# 保存的是字典，要新建模型
    new_vgg16.load_state_dict(torch.load("model/vgg16_m2.pth"))

    print(new_vgg16)



