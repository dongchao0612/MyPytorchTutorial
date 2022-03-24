import torch.nn as nn
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    input_data = torch.tensor(
        [[1, 2, 0, 3, 1],
         [0, 1, 2, 3, 1],
         [1, 2, 1, 0, 0],
         [5, 2, 3, 1, 1],
         [2, 1, 0, 1, 1]]
    )
    kernel = torch.tensor(
        [[1, 2, 1],
         [0, 1, 0],
         [2, 1, 0]]
    )
    print(input_data.shape, kernel.shape)
    input_data = torch.reshape(input_data, shape=(1, 1, 5, 5))
    kernel = torch.reshape(kernel, shape=(1, 1, 3, 3))
    print(input_data.shape, kernel.shape)
    output_data1 = F.conv2d(input=input_data, weight=kernel, stride=1)
    output_data2 = F.conv2d(input=input_data, weight=kernel, stride=2)
    output_data3 = F.conv2d(input=input_data, weight=kernel, stride=1, padding=1)
    output_data4 = F.conv2d(input=input_data, weight=kernel, stride=2, padding=3)
    print(output_data1)
    print(output_data2)
    print(output_data3)
    print(output_data4)
    print(output_data1.shape)
    print(output_data2.shape)
    print(output_data3.shape)
    print(output_data4.shape)

