import torch 
import torch.nn as nn
import torch.nn.functional as F

from arch.network import *

def create_model(name, num_class):

    # Baseline 1
    if name == "Framewise_Cross":
        model = Framewise_Cross(num_classes=num_class)

    if name == "Framewise_ArcFace":
        model = Framewise_ArcFace(num_classes=num_class)
    
    if name == "Framewise_MagFace":
        model = Framewise_MagFace(num_classes=num_class)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters) 

    return model