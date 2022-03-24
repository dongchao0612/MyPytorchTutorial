import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):  # 初始化
        super(Model, self).__init__()  # 父类初始化

    def forward(self, intput):
        output = intput + 1
        return output


if __name__ == '__main__':
    m = Model()
    x = torch.tensor(1.0)
    out = m(x)
    print(out)
