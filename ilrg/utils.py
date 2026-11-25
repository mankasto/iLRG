"""General helpers: settings objects, running averages, seeds, devices and log files."""

import os
import re
import torch
import random
import numpy as np

# MultiStepLR drops the learning rate at about 3/8, 5/8 and 7/8 of a run
LR_DECAY_DIVISORS = (2.667, 1.6, 1.142)


class Config:
    """Attribute access to a dict of settings."""

    def __init__(self, entries=None):
        self.__dict__.update(entries or {})


class AverageMeter(object):
    """Running average that keeps every value, for trimmed means (filter)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.bak = []

    def update(self, val, n=1):
        self.bak.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def filter(self, ratio):
        """Replace avg by the mean without the `ratio` share of smallest and of largest values."""
        self.avg = get_normal_mean(self.bak, ratio)


def get_normal_mean(data, ratio=0.00):
    data = np.array(data)
    data = np.sort(data)
    filters = int(len(data) * ratio)
    data = data[filters:len(data) - filters]
    return np.mean(data)


def lr_milestones(total_steps):
    """Steps (epochs or iterations) at which the learning rate drops."""
    return [total_steps // divisor for divisor in LR_DECAY_DIVISORS]


def get_device(use_cuda=True):
    use_cuda = torch.cuda.is_available() and use_cuda
    device = 'cuda' if use_cuda else 'cpu'
    return device


def set_random_seed(seed=233):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


def get_log_id(res_dir='results/runs/Experiment'):
    if not os.path.exists(res_dir):
        log_id = 1
    else:
        history_ids = [int(''.join(re.findall(r"\d+", log_path))) for log_path in os.listdir(res_dir)]
        log_id = max(history_ids) + 1 if len(history_ids) > 0 else 1
    return log_id


def get_log_file(log_dir):
    log_id = get_log_id(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    exp_log = os.path.join(log_dir, 'exp{}.log'.format(log_id))
    log_file = open(exp_log, 'w')
    return log_file


def get_params_string(params_dict):
    params_string = ''
    for k, v in params_dict.items():
        params_string += '\t{}: {}\n'.format(k, v)
    return params_string


def print_util(string, log_file):
    print(string)
    print(string, file=log_file)
