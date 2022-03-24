from PIL import Image
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


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
    img_path = "../img/dog.png"
    img = Image.open(img_path)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),
         torchvision.transforms.ToTensor()]
    )
    img = img.convert("RGB")
    img = transform(img)
    img = torch.reshape(img, (1, 3, 32, 32))
    print(img.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    new_model = torch.load("model/mymodule_39.pth")
    new_model.eval()
    with torch.no_grad():
        img = img.to(device)
        output = new_model(img)
    print(output)
    print(output.argmax(1))
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(classes[output.argmax(1)])
