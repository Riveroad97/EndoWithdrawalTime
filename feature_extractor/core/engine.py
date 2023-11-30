import math
import numpy as np
from monai.metrics import ROCAUCMetric
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import torch
import torch.nn.functional as F

from core.utils import *

# post processing
num_class = 4
auc_metric = ROCAUCMetric() # Average Macro

fn_tonumpy = lambda x: x.detach().cpu().numpy()

###### Baseline
def train_Framewise_Cross(model, data_loader, optimizer, device, epoch, config):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        
        image = batch_data[0].to(device).float()
        label = batch_data[1].to(device).long()
        
        pred = model(image)
        
        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label)
        else:
            loss = model.criterion(input=pred, target=label)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Framewise_Cross(model, data_loader, device, epoch,  config):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    header = 'Valid: [epoch:{}]'.format(epoch)

    label_list = []
    pred_list  = []

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image = batch_data[0].to(device).float()
        label = batch_data[1].to(device).long()

        pred = model(image)

        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label)
        else:
            loss = model.criterion(input=pred, target=label)
        loss_value = loss.item()

        metric_logger.update(loss=loss_value)
 
        # Post-processing
        y_onehot = F.one_hot(label, num_classes=config['num_class'])  # torch.Size([1, 3])
        y_pred_prob = F.softmax(pred, dim=1)           # torch.Size([1, 3])

        # Metric
        auc = auc_metric(y_pred=y_pred_prob, y=y_onehot) # [B, C]

        # Save
        pred_list.append(fn_tonumpy(pred.argmax(dim=1)).squeeze())
        label_list.append(fn_tonumpy(label).squeeze())
        
    # Metric
    label_list = np.concatenate(label_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)

    auc = auc_metric.aggregate()
    f1 = f1_score(y_true=label_list, y_pred=pred_list, average='macro')
    acc = accuracy_score(y_true=label_list, y_pred=pred_list)
    rec = recall_score(y_true=label_list, y_pred=pred_list, average='macro')
    pre = precision_score(y_true=label_list, y_pred=pred_list, average='macro')
    metric_logger.update(auc=auc, f1=f1, accuracy=acc, recall=rec, precision=pre)
    
    auc_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


###### ArcFace
def train_Framewise_ArcFace(model, data_loader, optimizer, device, epoch, config):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        
        image  = batch_data[0].to(device).float()
        label  = batch_data[1].to(device).long()
        
        pred = model(image)
        
        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label)
        else:
            loss = model.criterion(input=pred, target=label)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Framewise_ArcFace(model, data_loader, device, epoch,  config):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    header = 'Valid: [epoch:{}]'.format(epoch)

    label_list = []
    pred_list  = []

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image  = batch_data[0].to(device).float()
        label  = batch_data[1].to(device).long()

        pred= model(image)
        

        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label)
        else:
            loss = model.criterion(input=pred, target=label)
        loss_value = loss.item()
        pred = pred[0]

        metric_logger.update(loss=loss_value)
 
        # Post-processing
        y_onehot      = F.one_hot(label, num_classes=config['num_class'])  # torch.Size([1, 3])
        y_pred_prob   = F.softmax(pred, dim=1)           # torch.Size([1, 3])

        # Metric
        auc           = auc_metric(y_pred=y_pred_prob, y=y_onehot) # [B, C]

        # Save
        pred_list.append(fn_tonumpy(pred.argmax(dim=1)).squeeze())
        label_list.append(fn_tonumpy(label).squeeze())
        
    # Metric
    label_list = np.concatenate(label_list, axis=0)
    pred_list  = np.concatenate(pred_list, axis=0)

    auc  = auc_metric.aggregate()
    f1   = f1_score(y_true=label_list, y_pred=pred_list, average='macro')
    acc  = accuracy_score(y_true=label_list, y_pred=pred_list)
    rec  = recall_score(y_true=label_list, y_pred=pred_list, average='macro')
    pre  = precision_score(y_true=label_list, y_pred=pred_list, average='macro')
    metric_logger.update(auc=auc, f1=f1, accuracy=acc, recall=rec, precision=pre)
    
    auc_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


###### MagFace
def train_Framewise_MagFace(model, data_loader, optimizer, device, epoch, config):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image = batch_data[0].to(device).float()
        label = batch_data[1].to(device).long()
        
        pred, x_norm = model(image)
        
        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label, x_norm=x_norm)
        else:
            loss = model.criterion(input=pred, target=label, x_norm=x_norm)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid_Framewise_MagFace(model, data_loader, device, epoch,  config):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    header = 'Valid: [epoch:{}]'.format(epoch)

    label_list = []
    pred_list = []

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image = batch_data[0].to(device).float()
        label = batch_data[1].to(device).long()

        pred, x_norm = model(image)
        

        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label, x_norm=x_norm)
        else:
            loss = model.criterion(input=pred, target=label, x_norm=x_norm)
        loss_value = loss.item()
        pred = pred[0]

        metric_logger.update(loss=loss_value)
 
        # Post-processing
        y_onehot = F.one_hot(label, num_classes=config['num_class'])
        y_pred_prob = F.softmax(pred, dim=1)

        # Metric
        auc = auc_metric(y_pred=y_pred_prob, y=y_onehot)

        # Save
        pred_list.append(fn_tonumpy(pred.argmax(dim=1)).squeeze())
        label_list.append(fn_tonumpy(label).squeeze())
        
    # Metric
    label_list = np.concatenate(label_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)

    auc  = auc_metric.aggregate()
    f1 = f1_score(y_true=label_list, y_pred=pred_list, average='macro')
    acc = accuracy_score(y_true=label_list, y_pred=pred_list)
    rec = recall_score(y_true=label_list, y_pred=pred_list, average='macro')
    pre = precision_score(y_true=label_list, y_pred=pred_list, average='macro')
    metric_logger.update(auc=auc, f1=f1, accuracy=acc, recall=rec, precision=pre)
    
    auc_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



