import os
import time
import random
import datetime
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from matplotlib import *


## 네트워크 저장하기
def save(ckpt_dir, model, optimizer, lr_scheduler, epoch, config):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if config['gpu_mode'] == 'Single':
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict()},
                "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
    
    elif config['gpu_mode'] == 'DataParallel':
        torch.save({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict()},
                "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## 네트워크 불러오기
def load(ckpt_dir, model, optimizer, lr_scheduler):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return model, optimizer, lr_scheduler, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location='cpu')

    if 'module' in list(dict_model['model'].keys())[0]:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(dict_model['model'], strict=False)
    else:
        model.load_state_dict(dict_model['model'], strict=False)
        
    optimizer.load_state_dict(dict_model['optimizer'])
    lr_scheduler.load_state_dict(dict_model['lr_scheduler'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return model, optimizer, lr_scheduler, epoch


## log 읽기
def read_log(path):
    log_list = []
    lines = open(path, 'r').read().splitlines() 
    for i in range(len(lines)):
        exec('log_list.append('+lines[i] + ')')
    return  log_list


def seed_everything(seed):
    random.seed(seed)  # python random seed 고정
    os.environ['PYTHONHASHSEED'] = str(seed)  # os 자체의 seed 고정
    np.random.seed(seed)  # numpy seed 고정
    torch.manual_seed(seed)  # torch seed 고정
    torch.cuda.manual_seed(seed)  # cudnn seed 고정
    torch.backends.cudnn.deterministic = True  # cudnn seed 고정(nn.Conv2d)
    torch.backends.cudnn.benchmark = False  # CUDA 내부 연산에서 가장 빠른 알고리즘을 찾아 수행


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        # n is batch_size
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", n=1):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter  = delimiter
        self.n = n

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(value=v, n=self.n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)


    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def segment_bars_with_confidence_score(save_path, confidence_score, labels=[]):
    num_pics = len(labels)
    color_map = plt.cm.tab10


    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=3)
    fig = plt.figure(figsize=(15, (num_pics+1) * 1.5))

    interval = 1 / (num_pics+2)
    axes = []
    for i, label in enumerate(labels):
        i = i + 1
        axes.append(fig.add_axes([0.1, 1-i*interval, 0.8, interval - interval/num_pics]))
    titles = ['Ground Truth','Causal-TCN + MS-GRU','Causal-TCN', 'Causal-TCN + PKI']
    for i, label in enumerate(labels):
        label = [i for i in label]
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].imshow([label], **barprops)
        axes[i].set_title(titles[i])
    
    ax99 = fig.add_axes([0.1, 0.05, 0.8, interval - interval/num_pics])
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