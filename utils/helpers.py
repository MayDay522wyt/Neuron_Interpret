import pickle
import torchvision.transforms as transforms
from PIL import Image
import os
import re

def save_array(array, filename):
    open_file = open(filename, "wb")
    pickle.dump(array, open_file)
    open_file.close()

def read_masks(mask_path):
    #读取文件
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256像素
        transforms.CenterCrop(224),  # 中心裁剪为224x224像素
        transforms.ToTensor(),  # 转换为张量
    ])
    mask_files = os.listdir(mask_path)
    mask_files = sorted(mask_files)
    mask_tensor_list = []
    mask_name_list = []
    for mask_file in mask_files:
        mask = Image.open(os.path.join(mask_path,mask_file)).convert('L')
        mask_tensor = preprocess(mask)
        mask_tensor_list.append(mask_tensor)
        mask_name_list.append(mask_file.replace("-mask.jpg", ""))
    
    return mask_name_list, mask_tensor_list
