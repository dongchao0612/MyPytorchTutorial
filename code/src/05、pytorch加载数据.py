from torch.utils.data import Dataset
from PIL import Image
import os
class MyDataset(Dataset):
    def __init__(self,root_dir, lable_dir):
        self.root_dir=root_dir #../dataset/train
        self.lable_dir=lable_dir #ants
        self.path=os.path.join(self.root_dir,self.lable_dir) #../dataset/train\ants
        self.img_path=os.listdir(self.path)# 列表



    def __getitem__(self, index):
        self.name=self.img_path[index] #文件名列表
        self.img_item_path=os.path.join(self.path,self.name)#../dataset/train\ants\0013035.jpg
        self.img=Image.open(self.img_item_path)
        return self.img, self.lable_dir
    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    root_dir= r"../dataset/train"
    lable_dir_ants="ants"
    lable_dir_bees = "bees"
    ants_dataset=MyDataset(root_dir,lable_dir_ants)
    bees_dataset = MyDataset(root_dir, lable_dir_bees)
    img, lable_dir=ants_dataset.__getitem__(0)
    #print(ants_dataset.__len__())  # 121
    #print(bees_dataset.__len__())  # 124
