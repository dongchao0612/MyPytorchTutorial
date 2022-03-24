from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import  cv2 as cv
if __name__ == '__main__':
    # tensorboard --logdir=runs 在 dos下面进入code目录
    writer = SummaryWriter("runs")
    #添加数字
    # for i in range(100):
    #     writer.add_scalar(tag="y=2*x", scalar_value=2*i, global_step=i)
    #添加图片
    #img = Image.open("../dataset/train/ants/0013035.jpg") #<class 'PIL.JpegImagePlugin.JpegImageFile'>
    img=cv.imread("../dataset/train/ants/0013035.jpg") # <class 'numpy.ndarray'>
    writer.add_image(tag="ants", img_tensor=img, global_step=1, dataformats="HWC")
    writer.close()