import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

if __name__ == '__main__':
    input_data = torch.tensor([1, 2, 3], dtype=torch.float32)
    input_data = torch.reshape(input_data, (1, 1, 1, 3))

    target_data = torch.tensor([1, 2, 5], dtype=torch.float32)
    target_data = torch.reshape(target_data, (1, 1, 1, 3))

    loss = L1Loss(reduction="mean")
    loss_mse = MSELoss()  # 均方误差

    result = loss(input_data, target_data)
    result_mse = loss_mse(input_data, target_data)

    # print(result, result_mse)

    x = torch.tensor([0.1, 0.2, 0.3])
    y = torch.tensor([1])
    x = torch.reshape(x, (1, 3))

    loss_cross = CrossEntropyLoss()
    result_cross = loss_cross(x, y)
    print(result_cross)
