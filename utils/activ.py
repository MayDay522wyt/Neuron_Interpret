import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_ubyte
import os
from tqdm import tqdm
import numpy as np
import random
from utils import models, nethook, helpers
random.seed(42)

def store_activs(model, layernames):
    '''Store the activations in a list.'''
    activs = []
    for layer in layernames:
        activation = model.retained_layer(layer, clear = True)
        activs.append(activation)
        
    return activs


def dict_layers(activs):
    '''Return dictionary of layer sizes.'''
    all_layers = {}
    for iii, activ in enumerate(activs):
        all_layers[activs[iii]] = activ.shape[1]
    return all_layers

def get_image_activs(model_name, image_path:str="/export/home/wuyueting/nature_data/ImageNet-1k/train/n01558993/n01558993_10514.JPEG",
                     mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],
                     visualize=False, save=True):
    alpha = 0.003

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_layers = models.load(model_name, device)
    model.eval()

    ## hook layers for model
    model = nethook.InstrumentedModel(model)
    model.retain_layers(model_layers)

    # load image
    image = Image.open(image_path)
    
    if image.mode !='RGB':
        image = image.convert('RGB') 

    # 定义预处理的转换
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256像素
        transforms.CenterCrop(224),  # 中心裁剪为224x224像素
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=mean, std=std)  # 标准化
    ])

    # 应用预处理转换
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # 添加批次维度

    img_viz = torch.permute(input_batch[0].cpu(), (1,2,0))
    img_viz = ((img_viz-torch.min(img_viz))/(torch.max(img_viz)-torch.min(img_viz))).numpy()
    
    output= model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    # print(predicted_idx)

    #### append discriminator layer activations for batch
    model_activs =  store_activs(model, model_layers)
    if save == True:
        helpers.save_array(model_activs, os.path.join("activs_stats", "model_activs.pkl"))
    if visualize==True:
        for layer in range(len(model_activs)):
            os.makedirs("Layer"+str(layer), exist_ok=True)
            total_channel = model_activs[layer].shape[1]
            for channel in tqdm(range(total_channel)):
                fig=plt.figure(figsize=(13, 5))
                plt.axis("off")
                plt.title("Visualize Neuron -- Layer " +str(layer)+"  Channel "+str(channel), y=-0.1)
                
                model_act_viz = model_activs[layer][0,channel].unsqueeze(0).unsqueeze(0)
                model_act_viz = torch.nn.Upsample(size=(input_batch.shape[2], input_batch.shape[3]), mode='nearest')(model_act_viz).cpu()
                model_act_viz = (model_act_viz-torch.min(model_act_viz))/(torch.max(model_act_viz)-torch.min(model_act_viz))
                model_act_viz = img_as_ubyte(model_act_viz)
                model_act_viz = cv2.applyColorMap(model_act_viz[0][0], cv2.COLORMAP_JET)

                

                # minifig= fig.add_subplot(1, 3, 1)
                # minifig.axis('off')
                # minifig.title.set_text("Original Image")
                
                # plt.imshow(img_viz)

                minifig2 = fig.add_subplot(1, 2, 1)
                minifig2.axis('off')
                minifig2.title.set_text("Image Activation")
                plt.imshow(alpha*model_act_viz+img_viz)

                minifig3 = fig.add_subplot(1, 2, 2)
                minifig3.axis('off')
                minifig3.title.set_text("Activation Map")
                plt.imshow(model_activs[layer][0,channel].cpu())

                plt.savefig("Layer"+str(layer)+"/Channel"+str(channel)+".png") 

    return model_activs

def get_one_neuron_image_activs(model_name, layer, channel, image_path:str="/export/home/wuyueting/nature_data/ImageNet-1k/train/n01558993/n01558993_10514.JPEG",visualize=False,save_path = "my_plot.png"):
    alpha = 0.003

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_layers = models.load(model_name, device)
    model.eval()

    ## hook layers for model
    model = nethook.InstrumentedModel(model)
    model.retain_layers(model_layers)

    # load image
    image = Image.open(image_path)
    if image.mode !='RGB':
        image = image.convert('RGB') 

    # 定义预处理的转换
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256像素
        transforms.CenterCrop(224),  # 中心裁剪为224x224像素
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 应用预处理转换
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # 添加批次维度

    img_viz = torch.permute(input_batch[0].cpu(), (1,2,0))
    img_viz = ((img_viz-torch.min(img_viz))/(torch.max(img_viz)-torch.min(img_viz))).numpy()
    
    _ = model(input_batch)
    #### append discriminator layer activations for batch
    model_activs =  store_activs(model, model_layers)
    activ = model_activs[layer][0,channel].cpu()
    
    if visualize:
        model_act_viz = model_activs[layer][0,channel].unsqueeze(0).unsqueeze(0)
        model_act_viz = torch.nn.Upsample(size=(input_batch.shape[2], input_batch.shape[3]), mode='nearest')(model_act_viz).cpu()
        model_act_viz = (model_act_viz-torch.min(model_act_viz))/(torch.max(model_act_viz)-torch.min(model_act_viz))
        model_act_viz = img_as_ubyte(model_act_viz)
        model_act_viz = cv2.applyColorMap(model_act_viz[0][0], cv2.COLORMAP_JET)

        activ_image = alpha*model_act_viz+img_viz
        activ = model_activs[layer][0,channel].cpu()

        fig=plt.figure(figsize=(13, 5))
        plt.axis("off")
        plt.title("Visualize Neuron -- Layer " +str(layer)+"  Channel "+str(channel), y=-0.1)
        
        # minifig= fig.add_subplot(1, 3, 1)
        # minifig.axis('off')
        # minifig.title.set_text("Original Image")
        # plt.imshow(img_viz)

        minifig2 = fig.add_subplot(1, 2, 1)
        minifig2.axis('off')
        minifig2.title.set_text("Image Activation")
        plt.imshow(activ_image)

        minifig3 = fig.add_subplot(1, 2, 2)
        minifig3.axis('off')
        minifig3.title.set_text("Activation Map")
        plt.imshow(activ)

        # plt.savefig("Layer"+str(layer)+"/Channel"+str(channel)+".png") 
        plt.savefig(save_path) 
    

    return activ


def main(model_name,Layer, channel, image_folder="/export/home/wuyueting/nature_data/ImageNet-1k/train/n01558993/",k:int=9):
    image_names = os.listdir(image_folder)

    k_samples =  random.sample(image_names, k)

    image_activs = []
    original_image = []
    activs = []

    # 将图像平铺在画布上
    for i in range(9):
        activ_image, activ, img_viz = get_one_neuron_image_activs(model_name,Layer ,channel ,image_path=os.path.join(image_folder,k_samples[i]))
        image_activs.append(activ_image)
        original_image.append(img_viz)
        activs.append(activ)
    
    fig = plt.figure(figsize=(8, 8))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(original_image[i])
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.savefig(str(Layer)+"-"+str(channel)+"-image.png") 

    fig1 = plt.figure(figsize=(8, 8))
    for i in range(9):
        ax = fig1.add_subplot(3, 3, i+1)
        ax.imshow(activs[i])
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.savefig(str(Layer)+"-"+str(channel)+"-activ.png") 

    fig2 = plt.figure(figsize=(8, 8))
    for i in range(9):
        ax = fig2.add_subplot(3, 3, i+1)
        ax.imshow(image_activs[i])
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.savefig(str(Layer)+"-"+str(channel)+"-imgact.png") 

    

    return 