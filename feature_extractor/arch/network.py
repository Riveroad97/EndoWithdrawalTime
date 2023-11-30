from __future__ import print_function

import math
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F

from resnest.torch import resnest50


class Resnest50_Encoder(nn.Module):
    def __init__(self):
        super(Resnest50_Encoder, self).__init__()
        resnet = resnest50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
    
    def forward(self, x):
        x = self.share.forward(x)
        return x


class ArcLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, m=0.5, easy_margin=True):
        super(ArcLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = Parameter(torch.Tensor(in_features, out_features))
        self.scale        = scale            # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.m = m
        self.easy_margin  = easy_margin
        
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.eps = 1e-4

    def forward(self, x):
        weight_norm = F.normalize(self.weight, dim=0)
        cos_m, sin_m = torch.cos(self.m), torch.sin(self.m)

        cos_theta   = torch.mm(F.normalize(x), weight_norm)
        cos_theta   = cos_theta.clamp(-1+self.eps, 1-self.eps) # for stability
        sin_theta   = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm          = torch.sin(math.pi - self.m) * self.m
            threshold   = torch.cos(math.pi - self.m)
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)
            
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta   = self.scale * cos_theta

        return [cos_theta, cos_theta_m]


class ArcLoss(nn.Module):
    """
    ArcFace Loss.
    """

    def __init__(self):
        super(ArcLoss, self).__init__()
    

    def forward(self, input, target):
        cos_theta, cos_theta_m = input
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')
        
        loss = torch.mean(loss)
        return loss


class MagLinear(nn.Module):
    """
    Parallel fc for Mag loss
    """

    def __init__(self, in_features, out_features, scale=64.0, easy_margin=True):
        super(MagLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = Parameter(torch.Tensor(in_features, out_features))
        self.scale        = scale            # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.easy_margin  = easy_margin

        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, m, l_a, u_a):
        """
        Here m is a function which generate adaptive margin
        """
        x_norm       = torch.norm(x, dim=1, keepdim=True).clamp(l_a, u_a)
        ada_margin   = m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta   = torch.mm(F.normalize(x), weight_norm)
        cos_theta   = cos_theta.clamp(-1, 1)  # for numerical stability
        sin_theta   = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm          = torch.sin(math.pi - ada_margin) * ada_margin
            threshold   = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)
            
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta   = self.scale * cos_theta

        return [cos_theta, cos_theta_m], x_norm
    

class MagLoss(nn.Module):
    """
    MagFace Loss.
    """

    def __init__(self, l_a, u_a, l_margin, u_margin, scale=64.0):
        super(MagLoss, self).__init__()
        self.l_a = l_a
        self.u_a = u_a
        self.scale = scale
        self.cut_off = np.cos(np.pi/2-l_margin)
        self.large_value = 1 << 10
        self.lambda_g = 35 

    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def forward(self, input, target, x_norm):
        loss_g = self.calc_loss_G(x_norm)

        cos_theta, cos_theta_m = input
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')
        
        loss = torch.mean(loss) + self.lambda_g * loss_g
        return loss


# Cross-Entropy
class Framewise_Cross(nn.Module):
    def __init__(self, num_classes=4):
        super(Framewise_Cross, self).__init__()

        # Backbone
        self.backbone = Resnest50_Encoder()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

        self.initialize_classifier()

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    
    def initialize_classifier(self):
        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
    

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_feat(self, x):
        x = self.backbone(x)
        return x
    

# ArcFace
class Framewise_ArcFace(nn.Module):
    def __init__(self, num_classes=4):
        super(Framewise_ArcFace, self).__init__()

        # Backbone
        self.backbone = Resnest50_Encoder()

        self.fc = ArcLinear(in_features=2048, out_features=num_classes, scale=64)
    
        # Loss
        self.criterion = ArcLoss()
    
    def forward(self, x):
        x = self.backbone(x)
        logits = self.fc(x)
        return logits

    def extract_feat(self, x):
        x = self.backbone(x)
        return x


# MagFace
class Framewise_MagFace(nn.Module):
    def __init__(self, num_classes=4):
        super(Framewise_MagFace, self).__init__()

        # Backbone
        self.backbone = Resnest50_Encoder()

        self.fc = MagLinear(in_features=2048, out_features=num_classes, scale=64)
        self.l_margin = 0.4
        self.u_margin = 0.80
        self.l_a = 10
        self.u_a = 110
    
        # Loss
        self.criterion = MagLoss(l_a=self.l_a, u_a=self.u_a, l_margin=self.l_margin, u_margin=self.u_margin)
    
    def _margin(self, x):
        # generate adaptive margin
        margin = (self.u_margin-self.l_margin) / (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin

    
    def forward(self, x):
        x = self.backbone(x)
        logits, x_norm = self.fc(x, self._margin, self.l_a, self.u_a)
        return logits, x_norm

    def extract_feat(self, x):
        x = self.backbone(x)
        return x





    

