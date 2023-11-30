import os
import sys
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import *

import torch
import torch.nn.functional as F
import torch.nn as nn


phase2label_dicts = {
    'cecum3':{
    'cecum':0,
    'background':1,
    'out':2
    },
    'cecum4':{
    'cecum':0,
    'background':1,
    'out':2,
    'surgery':3
    }
}

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] for label in labels]
    return phases

max_pool = nn.MaxPool1d(kernel_size=13,stride=5,dilation=3)

def fusion(predicted_list,labels,args):

    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0
    len_layer = len(predicted_list)
    weight_list = [1.0/len_layer for i in range (0, len_layer)]

    for out, w in zip(predicted_list, weight_list):
        resize_out =F.interpolate(out,size=labels.size(0),mode='nearest')
        resize_out_list.append(resize_out)

        resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0),size=out.size(2),mode='linear',align_corners=False)
        if out.size(2)==labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
          
            resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0),size=out.size(2),mode='nearest')
          
            labels_list.append(resize_label.squeeze().long())


        all_out_list.append(out)


    return all_out, all_out_list, labels_list


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    # dist = 1. - similiarity
    return similiarity


def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.cm.tab10
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=3)
    
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def segment_bars_with_confidence_score(save_path, confidence_score, labels=[]):
    num_pics = len(labels)
    # color_map = plt.cm.Set1
    color_map = plt.cm.tab10

#     axprops = dict(xticks=[], yticks=[0,0.5,1], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=3)
    fig = plt.figure(figsize=(15, (num_pics+1) * 1.5))

    interval = 1 / (num_pics+2)
    axes = []
    for i, label in enumerate(labels):
        i = i + 1
        axes.append(fig.add_axes([0.1, 1-i*interval, 0.8, interval - interval/num_pics]))
#         ax1.imshow([label], **barprops)
    titles = ['Ground Truth','Causal-TCN + MS-GRU','Causal-TCN', 'Causal-TCN + PKI']
    for i, label in enumerate(labels):
        label = [i for i in label]
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].imshow([label], **barprops)
        axes[i].set_title(titles[i])
    
    ax99 = fig.add_axes([0.1, 0.05, 0.8, interval - interval/num_pics])
#     ax99.set_xlim(-len(confidence_score)/15, len(confidence_score) + len(confidence_score)/15)
    ax99.set_xlim(0, len(confidence_score))
    ax99.set_ylim(-0.2, 1.2)
    ax99.set_yticks([0,0.5,1])
    ax99.set_xticks([])
    ax99.set_title('Confidence Score')
 
     
    ax99.plot(range(len(confidence_score)), confidence_score)

    if save_path is not None:
        print(save_path)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    

def PKI(confidence_seq, prediction_seq, transition_prior_matrix, alpha, beta, gamma): # fix the predictions that do not meet priors
    initital_phase = 0
    previous_phase = 0
    alpha_count = 0
    assert len(confidence_seq) == len(prediction_seq)
    refined_seq = []
    for i, prediction in enumerate(prediction_seq):
        if prediction == initital_phase:
            alpha_count = 0
            refined_seq.append(initital_phase)
        else:
            if prediction != previous_phase or confidence_seq[i] <= beta:
                alpha_count = 0
            
            if confidence_seq[i] >= beta:
                alpha_count += 1
            
            if transition_prior_matrix[initital_phase][prediction] == 1:
                refined_seq.append(prediction)
            else:
                refined_seq.append(initital_phase)
            
            if alpha_count >= alpha and transition_prior_matrix[initital_phase][prediction] == 1:
                initital_phase = prediction
                alpha_count = 0
                
            if alpha_count >= gamma:
                initital_phase = prediction
                alpha_count = 0
        previous_phase = prediction

    assert len(refined_seq) == len(prediction_seq)
    return refined_seq


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues, save_path=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()