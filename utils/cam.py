import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import torch.nn as nn

import argparse
import cv2
import numpy as np
import torch
import os 

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='cam_output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args

def reshape_transform(tensor, height=16, width=16):  
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class CLIPModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(CLIPModelOutputWrapper, self).__init__()
        self.model = model
    
    def forward(self, image_inputs):
        return self.model.get_image_features(image_inputs)
    
class CLIPModelTarget:
    def __init__(self,text_features):
        self.text_features = text_features
    
    
    def __call__(self, image_features):
        image_features = image_features/ image_features.norm(p=2, dim=-1, keepdim=True)
        self.text_features = self.text_features/ self.text_features.norm(p=2, dim=-1, keepdim=True)
        logits_per_text = torch.matmul(self.text_features, image_features.t()) 
        return logits_per_text

def cam(model, processor, tokenizer, image_path, concept: str, save=False):
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    
    device = args.device

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = model.to(device)
    # model = CLIPModel.from_pretrained(model_path).to(device)
    # processor = CLIPProcessor.from_pretrained(model_path)
    # # print(processor)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    # rgb_img = np.float32(rgb_img) / 255
    # print(rgb_img.shape)
    mean  = [0.48145466, 0.4578272, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    image = Image.open(image_path)
    image_inputs = processor(images=image, return_tensors="pt")['pixel_values']
    rgb_img = image_inputs.permute(0,2,3,1)
    rgb_img = rgb_img.numpy()
    rgb_img = np.squeeze(rgb_img)
    rgb_img = rgb_img*std + mean

    inputs = tokenizer(["a photo of "+concept], padding=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(**inputs)

    model = CLIPModelOutputWrapper(model).eval()

    target_layers = [model.model.vision_model.encoder.layers[-1].layer_norm1]
    targets = [CLIPModelTarget(text_features)]

    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       reshape_transform=reshape_transform) as cam:
                       
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=image_inputs,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        if save:
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            cv2.imwrite(os.path.join(args.output_dir,f'{args.method}_{concept}_cam.jpg'), cam_image)
    return grayscale_cam

