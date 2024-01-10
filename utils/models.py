import torch
from torchvision.models import resnet50
import os
from mae import load_mae
import models_vit

def prepare_model(chkpt_dir="/export/home/wuyueting/Encoder/mae_like/mae_main_wyt_revised/output_dir/large/i_lcond/finetune/0.85/large_lcond_finetune_85_bestauc.pth", arch="vit_large_patch16"):
    # build model
    model = models_vit.__dict__[arch](
        num_classes=2,
        global_pool=True,
    )
    # load model
    if chkpt_dir != None:
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
    return model

def load(model_name, device="cpu", path:str = "/export/home/wuyueting/Interpretability/rosetta_neurons-main"):
    if model_name == "resnet50":
        model = resnet50(weights="IMAGENET1K_V2").to(device)
        model_layers = [ "layer1", "layer2", "layer3", "layer4"]
        for p in model.parameters(): 
            p.data = p.data.float()
    if model_name == 'mae':
        model = load_mae(os.path.join(path, '/export/home/wuyueting/Interpretability/rosetta_neurons-main/mae_pretrain_vit_base.pth')).to(device)
        #discr_layers = [f"blocks.{i}" for i in range(12)]
        model_layers = []
        for name, layer in model.named_modules():
            if  "mlp.act" in name:
                model_layers.append(name)
    if model_name == 'thyroid_vit':
        model = prepare_model().to(device)
        model_layers = []
        for name, layer in model.named_modules():
            if  "mlp.act" in name:
                model_layers.append(name)

    return model, model_layers

if __name__ =="__main__":
    load("mae")