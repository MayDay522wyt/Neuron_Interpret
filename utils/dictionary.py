from utils import cam
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

def make_concept_dic(model, processor, tokenizer, image_path, concepts, save=False):
    dictionary = {}

    for concept in concepts:
        grayscale_cam = cam.cam(model, processor, tokenizer, image_path, concept, save)
        dictionary[concept] = grayscale_cam
        del grayscale_cam
        
    return dictionary

if __name__ == "__main__":
    model_path = "/export/home/wuyueting/CLIP-like/hf-CLIP/clip-vit-large-patch14"
    concepts = ["face", "body", "foot", "eyes", "grass", "person", "background",]
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dictionary = make_concept_dic(model, processor, tokenizer, "/export/home/wuyueting/nature_data/PartImageNet_OOD/val/n02099601/n02099601_10705.JPEG", concepts, save=True)
    print(dictionary.keys())