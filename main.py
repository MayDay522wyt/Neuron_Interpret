import torch
from utils import models, nethook, activ, matching, helpers, dictionary
from neuron_explainer import explainer
import pickle
import os 
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import random


def main(model_name):
    nature = True
    # activ.get_image_activs(model_name, image_path = "/export/home/wuyueting/thyroid_data/annotated_data/image/002569548_20171026_US_1_9_9.jpg",mean=0.22, std=0.08)
    # all_match_table = matching.component_matching("/export/home/wuyueting/thyroid_data/annotated_data/mask/002569548_20171026_US_1_9_9_nodule.jpg", "activs_stats/model_activs.pkl",matching_score=50)
    # with open('activs_stats/table.pkl',"rb") as f:
    #     all_match_table = pickle.load(f)
    # # all_match_table = all_match_table.unsqueeze(0)
    # print(all_match_table.shape)
    # # last_layer = all_match_table[:, 11*3072:]
    # _, indexs = torch.topk(all_match_table, k=5, dim=0)
    # print(indexs)
    # for j in range(len(indexs)):
    #     index = indexs[j]
    #     for i in index:
    #         layer = i.numpy()//4096
    #         channel = int(i - 4096*layer)
    #         os.makedirs("matching_result/best_thyroid/",exist_ok=True)
    #         print(layer,channel)
    #         activ.get_one_neuron_image_activs(model_name,layer=layer, channel=channel,image_path="/export/home/wuyueting/thyroid_data/annotated_data/image/002569548_20171026_US_1_9_9.jpg",visualize=True,save_path ="matching_result/best_thyroid/"+str(layer)+"-"+str(channel)+".jpg")
    if nature==False:
        image_path = "/export/home/wuyueting/thyroid_data/annotated_data/image/002569548_20171026_US_1_9_9.jpg"
        activation = activ.get_one_neuron_image_activs(model_name, 18, 3500, image_path)
        
        sample = 1
        if sample == 1:
            model_path = "/export/home/wuyueting/CLIP-like/hf-CLIP/clip-vit-large-patch14"
            concepts = ["echo","composition", "edges","shape"]
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            concept_dict = dictionary.make_concept_dic(model, processor, tokenizer, image_path, concepts)
            scores = matching.matching_one_neuron(activation, concept_dict)
            print(scores)

        if sample == 2:
            matching_list = []
            images_names = os.listdir("/export/home/wuyueting/thyroid_data/train_old/image/")
            images_names = random.sample(images_names, 5)
            model_path = "/export/home/wuyueting/CLIP-like/hf-CLIP/clip-vit-large-patch14"
            concepts =["circle", "boundary", "hole", "nodule", "cell", "background"]
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            for image_name in images_names:
                # sub_class = image_name.split('_')[0]
                file_path = os.path.join("/export/home/wuyueting/thyroid_data/train_old/image/",image_name)
                activation = activ.get_one_neuron_image_activs(model_name, 16, 3465, image_path =file_path)
                # names, masks = helpers.read_masks(os.path.join("matching_data/train", image_name))
                # scores = matching.matching_one_neuron(activation, names, masks)
                concept_dict = dictionary.make_concept_dic(model, processor, tokenizer, file_path, concepts)
                scores = matching.matching_one_neuron(activation, concept_dict)
                matching_list.append(scores)

            # for note in matching_list:
            #     print(note)
            prompt = explainer.make_explanation_prompt(matching_list)
            print(prompt)
    
    
    activ.get_image_activs(model_name, image_path = "/export/home/wuyueting/nature_data/PartImageNet_OOD/val/n02009229/n02009229_11045.JPEG", visualize=False)
    # model,model_layer = models.load("mae")
    # print(model)
    # print(model_layer)

    # names, all_match_table = matching.components_matching("data_mask", "activs_stats/model_activs.pkl")
    # print(all_match_table)
    # with open('activs_stats/10table.pkl',"rb") as f:
    #     all_match_table = pickle.load(f)
    # print(all_match_table.shape)
    # # last_layer = all_match_table[:, 11*3072:]
    # _, indexs = torch.topk(all_match_table, k=1, dim=1)
    # for j in range(len(indexs)):
    #     index = indexs[j]
    #     for i in index:
    #         layer = i.numpy()//3072
    #         channel = int(i - 3072*layer)
    #         os.makedirs("matching_result/best/"+str(j),exist_ok=True)
    #         activ.get_one_image_activs(model_name,layer, channel,"/export/home/wuyueting/nature_data/PartImageNet_OOD/val/n02009229/n02009229_11045.JPEG","matching_result/best/"+str(j)+"/"+str(layer)+"-"+str(channel)+".jpg")
    # matching.matching(model_name)

    
    
    #
    if nature==True:
        image_path = "/export/home/wuyueting/nature_data/PartImageNet_OOD/val/n02009229/n02009229_11045.JPEG"
        activation = activ.get_one_neuron_image_activs(model_name, 6, 2046, image_path)
        # names, masks = helpers.read_masks('data_mask')
        # scores = matching.matching_one_neuron(activation, names, masks)

        sample = 2
        ## 对一个sample进行处理
        if sample == 1:
            model_path = "/export/home/wuyueting/CLIP-like/hf-CLIP/clip-vit-large-patch14"
            concepts = ["face", "body", "foot", "eyes", "grass", "background"]
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            concept_dict = dictionary.make_concept_dic(model, processor, tokenizer, image_path, concepts)
            scores = matching.matching_one_neuron(activation, concept_dict)
            print(scores)
        
        if sample == 2:
        # 对多个sample进行处理
            matching_list = []
            images_names = os.listdir("matching_data/train")
            model_path = "/export/home/wuyueting/CLIP-like/hf-CLIP/clip-vit-large-patch14"
            concepts = ["face", "body", "foot", "eyes", "grass", "background"]
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            for image_name in images_names:
                sub_class = image_name.split('_')[0]
                file_path = os.path.join("/export/home/wuyueting/nature_data/PartImageNet_OOD/val/",sub_class, image_name+".JPEG")
                activation = activ.get_one_neuron_image_activs(model_name, 6, 2046, image_path =file_path)
                # names, masks = helpers.read_masks(os.path.join("matching_data/train", image_name))
                # scores = matching.matching_one_neuron(activation, names, masks)
                concept_dict = dictionary.make_concept_dic(model, processor, tokenizer, file_path, concepts)
                scores = matching.matching_one_neuron(activation, concept_dict)
                matching_list.append(scores)

            # for note in matching_list:
            #     print(note)
            prompt = explainer.make_explanation_prompt(matching_list)
            print(prompt)
    
    
    return 0
    
if __name__ == "__main__":
    # main("mae")
    main('mae')