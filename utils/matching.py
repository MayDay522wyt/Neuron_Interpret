import pickle
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
from tqdm import tqdm
from utils import helpers, activ

def normalize(activation):
    # 把激活值标准化到0到1
    eps=0.0000001
    norm_activation = (activation-torch.min(activation))/(torch.max(activation)-torch.min(activation)+eps)
    
    return norm_activation

def create_final_table(all_match_table):
    num_mask = len(all_match_table)
    num_activs = len(all_match_table[0])
    final_match_table = torch.zeros(num_mask, num_activs)

    for i in range(num_mask):
        final_match_table[i,:] = torch.tensor(all_match_table[i])
    
    return final_match_table

def components_matching(mask_path:str, activs_path:str,matching_score:int=5):
    # matching score是匹配上的patch的得分，在使用”面“为主要形式的mask时，matching_score可以选为5，在使用以“线”为主导的mask时，matching_score可以选大一点的，比如50
    # a mask dir
    
    #读取文件
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256像素
        transforms.CenterCrop(224),  # 中心裁剪为224x224像素
        transforms.ToTensor(),  # 转换为张量
    ])
    mask_files = os.listdir(mask_path)
    mask_tensor_list = []
    for mask_file in mask_files:
        mask = Image.open(os.path.join(mask_path,mask_file)).convert('L')
        mask_tensor = preprocess(mask)
        mask_tensor_list.append(mask_tensor)

    with open(activs_path,"rb") as f:
        model_activs = pickle.load(f)

    size = 224

    all_match_table = []
    for i in range(len(mask_tensor_list)):
        component = mask_tensor_list[i].unsqueeze(0).cpu()
        match_table = []
        for j in tqdm(range(len(model_activs))):
            layer_activs = model_activs[j][0]
            for k in tqdm(range(len(layer_activs))):
                ori_activ = layer_activs[k].unsqueeze(0).unsqueeze(0).cpu()
                new_activ = torch.nn.Upsample(size=(size,size), mode='bilinear')(ori_activ)

                #进行components和activs的归一化，使得他们分布在-1到1的区间内
                norm_new_activ = normalize(new_activ)
                norm_component = component * matching_score - 1
                # print(torch.max(norm_new_activ),torch.min(norm_new_activ))

                #Pearson Correlation 
                prod = torch.einsum('aixy,ajxy->ij', norm_component,norm_new_activ)
                div1 = torch.einsum('aixy->i', norm_component**2)
                div2 = torch.einsum('ajxy->j', norm_new_activ**2)
                div = torch.einsum('i,j->ij', div1,div2)
                scores = prod/torch.sqrt(div)
                nans = torch.isnan(scores)
                scores[nans] = 0
                
                match_table.append(scores)
                del new_activ
                del norm_new_activ
                del scores
                break
        all_match_table.append(match_table)
        del match_table
        
    final_match_table = create_final_table(all_match_table)
    helpers.save_array(final_match_table, os.path.join("activs_stats", "table.pkl"))
    
    return  mask_files, final_match_table

def component_matching(mask_path:str, activs_path:str,matching_score:int=5):
    # matching score是匹配上的patch的得分，在使用”面“为主要形式的mask时，matching_score可以选为5，在使用以“线”为主导的mask时，matching_score可以选大一点的，比如50
    
    # one mask file
    
    #读取文件
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256像素
        transforms.CenterCrop(224),  # 中心裁剪为224x224像素
        transforms.ToTensor(),  # 转换为张量
    ])

    mask = Image.open(mask_path).convert('L')
    mask_tensor = preprocess(mask)

    with open(activs_path,"rb") as f:
        model_activs = pickle.load(f)

    size = 224


    component = mask_tensor.unsqueeze(0).cpu()
    match_table = []
    for j in tqdm(range(len(model_activs))):
        layer_activs = model_activs[j][0]
        for k in tqdm(range(len(layer_activs))):
            ori_activ = layer_activs[k].unsqueeze(0).unsqueeze(0).cpu()
            new_activ = torch.nn.Upsample(size=(size,size), mode='bilinear')(ori_activ)

            #进行components和activs的归一化，使得他们分布在-1到1的区间内
            norm_new_activ = normalize(new_activ)
            norm_component = component * matching_score - 1
            # print(torch.max(norm_new_activ),torch.min(norm_new_activ))

            #Pearson Correlation 
            prod = torch.einsum('aixy,ajxy->ij', norm_component,norm_new_activ)
            div1 = torch.einsum('aixy->i', norm_component**2)
            div2 = torch.einsum('ajxy->j', norm_new_activ**2)
            div = torch.einsum('i,j->ij', div1,div2)
            scores = prod/torch.sqrt(div)
            nans = torch.isnan(scores)
            scores[nans] = 0
            
            match_table.append(scores)
            del new_activ
            del norm_new_activ
            del scores
        
    final_match_table = create_final_table(match_table)
    helpers.save_array(final_match_table, os.path.join("activs_stats", "table.pkl"))
    
    return  final_match_table

def matching(model_name):
    # 读取n个照片
    image_names = os.listdir("matching_data/train/")
    image_names = sorted(image_names)

    for epoch in range(len(image_names)):
        tqdm.write(f'Processing item: {epoch}')
        # print("Dealing image "+str(epoch)+"...")
        image_name = image_names[epoch]
        image_class = image_name.split("_")[0]
        image_path = os.path.join("/export/home/wuyueting/nature_data/PartImageNet_OOD/val",image_class,image_name+".JPEG")

        model_activs = activ.get_image_activs(model_name, image_path, save=False)
        
        #读取文件
        preprocess = transforms.Compose([
            transforms.Resize(256),  # 调整图像大小为256x256像素
            transforms.CenterCrop(224),  # 中心裁剪为224x224像素
            transforms.ToTensor(),  # 转换为张量
        ])
        mask_files = os.listdir("matching_data/train/"+image_name)
        mask_files = sorted(mask_files)
        mask_tensor_list = []
        for mask_file in mask_files:
            mask = Image.open(os.path.join("matching_data/train/"+image_name,mask_file)).convert('L')
            mask_tensor = preprocess(mask)
            mask_tensor_list.append(mask_tensor)

        size = 224

        all_match_table = []
        for i in tqdm(range(len(mask_tensor_list)), desc='Processing mask'):
            component = mask_tensor_list[i].unsqueeze(0).cpu()
            match_table = []
            for j in tqdm(range(len(model_activs)), desc='Processing neuron layer'):
                layer_activs = model_activs[j][0]
                for k in range(len(layer_activs)):
                    ori_activ = layer_activs[k].unsqueeze(0).unsqueeze(0).cpu()
                    new_activ = torch.nn.Upsample(size=(size,size), mode='bilinear')(ori_activ)

                    #进行components和activs的归一化
                    norm_new_activ = normalize(new_activ) #0到1
                    norm_component = component * 5 - 1#-1到4
                    # print(torch.max(norm_new_activ),torch.min(norm_new_activ))

                    #Pearson Correlation 
                    prod = torch.einsum('aixy,ajxy->ij', norm_component,norm_new_activ)
                    div1 = torch.einsum('aixy->i', norm_component**2)
                    div2 = torch.einsum('ajxy->j', norm_new_activ**2)
                    div = torch.einsum('i,j->ij', div1,div2)
                    scores = prod/torch.sqrt(div)
                    nans = torch.isnan(scores)
                    scores[nans] = 0
                    
                    match_table.append(scores)
                    del new_activ
                    del norm_new_activ
                    del scores
                    # break
            all_match_table.append(match_table)
            del match_table

        batch_match_table = create_final_table(all_match_table)
        
        if epoch == 0:
            final_match_table = torch.zeros((batch_match_table.shape[0], batch_match_table.shape[1]))

        final_match_table += batch_match_table
        del all_match_table
        del batch_match_table

    final_match_table /= len(image_names)
    # print(final_match_table)
    helpers.save_array(final_match_table, os.path.join("activs_stats", "10table.pkl"))

    return final_match_table


def matching_one_neuron(activ, dictionary, method:str="Pearson"):
    ###
    # Function: 通过将激活图和已定概念的掩码进行比对，得到激活图和概念的关系
    # activ: neuron的激活值
    # concept_mask: 已定概念concepts的掩码，针对的是和neuron相同的样本, 
    ###

    # ccps = ccp_masks.keys()
    # masks = ccp_masks.values() 
    # assert activ.shape[-1,-2] == masks.shape[-1, -2], "The activation and mask are not in the same shape"
    
    concept_scores={}
    scores = []
    ### Method 1 : 只看重复

    ### Method 2 : 主成分分析

    ### Method 3 : Pearson Correlation
    if method == "Pearson":
        for ccp in dictionary:
            mask = dictionary[ccp]
            score = matching_method_Pearson(activ, mask)
            scores.append(score)
            concept_scores[ccp] = score
        
    ### Method ...
    # dict = {ccp: coefficient}

    return concept_scores

def matching_method_Pearson(ori_activ, mask):
    size = mask.shape[-1]
    if len(ori_activ.shape) == 2:
        ori_activ = ori_activ.unsqueeze(0).unsqueeze(0)
    
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask)

    if len(mask.shape) == 3:
        mask = mask.unsqueeze(0)
    elif len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)

    new_activ = torch.nn.Upsample(size=(size,size), mode='bilinear')(ori_activ)
    #进行components和activs的归一化
    norm_new_activ = normalize(new_activ) #0到1

    #Pearson Correlation 
    prod = torch.einsum('aixy,ajxy->ij', mask,norm_new_activ)
    div1 = torch.einsum('aixy->i', mask**2)
    div2 = torch.einsum('ajxy->j', norm_new_activ**2)
    div = torch.einsum('i,j->ij', div1,div2)
    scores = prod/torch.sqrt(div)
    nans = torch.isnan(scores)
    scores[nans] = 0
    scores = scores[0,0].item()

    return scores # score in [-1, 1]