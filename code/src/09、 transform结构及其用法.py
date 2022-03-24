from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    img = Image.open("../dataset/train/ants/0013035.jpg")
    # 将PIL文件转化为Tensor
    #print(img)  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x23542745F70>
    #print(type(img))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    trans_totensor = transforms.ToTensor()
    #print(trans_forms, type(trans_forms))  # ToTensor() <class 'torchvision.transforms.transforms.ToTensor'>
    tensor_img = trans_totensor(img)
    #print(tensor_img, type(tensor_img))  # <class 'torch.Tensor'>
    writer = SummaryWriter("runs")

    writer.add_image(tag="ants", img_tensor=tensor_img)
    writer.close()

