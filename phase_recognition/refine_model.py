import copy
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineCausualTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(RefineCausualTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class RefineGRU(nn.Module):
    def __init__(self, num_f_maps, num_classes):
        super(RefineGRU, self).__init__()
        self.gru = nn.GRU(input_size=num_classes, hidden_size=num_f_maps, batch_first=True)
        self.fc = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(num_f_maps, num_classes)
                                )

    def forward(self, x):
        # x of shape (batch, seq, feature)
        out, hn = self.gru(x) # out of shape (batch, seq, hidden)
        out = self.fc(out) # out of shape (batch, seq, num_class)
        return out, hn
    

class RefineTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(RefineTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageRefineGRU(nn.Module):
    def __init__(self, num_stage, num_f_maps, num_classes):
        super(MultiStageRefineGRU, self).__init__()
        
        self.stage1 = RefineGRU(num_f_maps, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(RefineGRU(num_f_maps, num_classes)) for s in range(num_stage-1)])
 
    def forward(self, x):
        x = x.permute(0,2,1) # (b,c,l) -> (b,l,c)
        out, _ = self.stage1(x) # out of shape (b,l,c)
        outputs = out.permute(0,2,1).unsqueeze(0) #(b,l,c) -> (1, b,c,l)
        for s in self.stages:
            out, _ = s(F.softmax(out, dim=2).detach())
#             out, _ = s(F.softmax(out, dim=2))
            outputs = torch.cat((outputs, out.permute(0,2,1).unsqueeze(0)), dim=0) #(n, b, c, l)
        
        return outputs, _


class MultiStageRefineCausalTCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageRefineCausalTCN, self).__init__()
        self.stage1 = RefineCausualTCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(RefineCausualTCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
        
    def forward(self, x, mask=None):
        # x of shape (bs, c, l)
        if mask is not None:
            x = x * mask

        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1).detach()) # bs x c_in x l_in   
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs, None


class MultiStageRefineTCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageRefineTCN, self).__init__()
        self.stage1 = RefineTCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(RefineTCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
        
    def forward(self, x, mask=None):
        # x of shape (bs, c, l)
        if mask is not None:
            x = x * mask


        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1).detach()) # bs x c_in x l_in   
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs, None

    
class DilatedResidualCausalLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualCausalLayer, self).__init__()
        self.padding = 2 * dilation
        # causal: add padding to the front of the input
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation) #
        # self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.pad(x, [self.padding, 0], 'constant', 0) # add padding to the front of input
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)