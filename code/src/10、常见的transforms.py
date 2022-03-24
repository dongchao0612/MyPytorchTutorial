from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    img = Image.open("../dataset/train/ants/0013035.jpg") #<class 'PIL.JpegImagePlugin.JpegImageFile'>
    writer = SummaryWriter("runs")

    # ToTensor
    trans_totensor = transforms.ToTensor()
    tensor_img = trans_totensor(img) #<class 'torch.Tensor'>
    writer.add_image(tag="ToTensor", img_tensor=tensor_img)

    # Normalize
    trans_normal = transforms.Normalize(mean=[0.5, 0.5, 0.5],std= [3, 5, 7])
    normal_img = trans_normal(tensor_img)
    writer.add_image(tag="Normalize", img_tensor=normal_img)

    # Resize
    #print(img.size) #(768, 512)
    trans_resize = transforms.Resize((512, 512))
    img_resize = trans_resize(img)  # PIL image
    img_resize = trans_totensor(img_resize)#torch.Size([3, 512, 512])
    writer.add_image(tag="Resize", img_tensor=img_resize,global_step=1)

    # Compose
    trans_resize2 = transforms.Resize(512) # 列宽
    trans_compose = transforms.Compose([trans_resize2, trans_totensor])
    img_resize2 = trans_compose(img)
    writer.add_image(tag="Resize", img_tensor=img_resize2, global_step=2)

    # RandomCrop 随机裁剪
    trans_random = transforms.RandomCrop(512)
    trans_compose2 = transforms.Compose([trans_random, trans_totensor])
    for i in range(10):
        img_crop = trans_compose2(img)
        writer.add_image("randomcrop", img_crop, i)
    writer.close()


