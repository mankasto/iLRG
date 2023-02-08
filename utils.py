import os
import re
import torch
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from methods import iLRG, sim_iLRG, idlg_infer, gi_infer, svd_infer


class Config:
    def __init__(self, entries: dict = {}):
        self.__dict__.update(entries)


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.bak = []
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
        self.avg = get_normal_mean(self.bak, ratio)


def get_normal_mean(data, ratio=0.00):
    data = np.array(data)
    data = np.sort(data)
    filters = int(len(data) * ratio)
    data = data[filters:len(data) - filters]
    return np.mean(data)


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


def split_integer(prototype, part):
    if not (isinstance(prototype, int) and isinstance(part, int)):
        raise ValueError('Input params should be integer.')
    if part <= 1:
        raise ValueError('Split part should be more than 1.')
    if prototype <= part:
        raise ValueError('Split part should be more than split prototype.')
    board_set = set()
    while len(board_set) < part - 1:
        board_set.add(random.randrange(1, prototype))
    board_list = list(board_set)
    board_list.append(0)
    board_list.append(prototype)
    board_list.sort()
    return [board_list[i + 1] - board_list[i] for i in range(part)]


def get_data(dataset,
             num_images,
             num_classes,
             start_id=0,
             num_uniform_cls=5,
             num_target_cls=5,
             data_distribution='random',
             device='cpu'):
    images, labels = [], []
    if data_distribution == 'extreme':
        cnt = 0
        target_id = start_id
        extreme_class = random.randint(0, num_classes - 1)
        while cnt < num_images:
            image, label = dataset[target_id]
            if label == extreme_class:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cnt += 1
            target_id += 1
            target_id = target_id % len(dataset)
    elif data_distribution == 'random':
        idx_list = random.sample(range(len(dataset)), num_images)
        for idx in idx_list:
            image, label = dataset[idx]
            images.append(image.float().to(device))
            labels.append(torch.as_tensor((label,), device=device))
        target_id = idx_list[0]
    elif data_distribution == 'balanced':
        target_id = start_id
        uniform_clses = random.sample(range(num_classes), num_uniform_cls)
        num_per_cls = num_images // num_uniform_cls
        cls_cnt = {cls: 0 for cls in uniform_clses}
        while min(list(cls_cnt.values())) < num_per_cls:
            image, label = dataset[target_id]
            if label in uniform_clses and cls_cnt[label] < num_per_cls:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)
    elif data_distribution == 'random2':
        target_id = start_id
        target_clses = random.sample(range(num_classes), num_target_cls)
        # target_clses = random.sample([8, 9, 10, 11, 16, 20, 22, 28, 34, 40, 45, 53, 57, 58, 82, 85, 86, 98],
        #                              num_target_cls)
        # target_clses = random.sample([10, 20, 28, 31, 35, 40, 58, 61, 69, 82, 98, 0, 24],
        #                              num_target_cls)
        # target_clses = random.sample([20, 58, 40, 61, 10, 76, 24, 77, 7, 19, 25, 5, 75],
        #                              num_target_cls)
        random_num = split_integer(num_images, num_target_cls)
        cls_num = {target_clses[i]: random_num[i] for i in range(num_target_cls)}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'custom_imbalanced':
        target_id = start_id
        target_clses = [0, 18, 92]
        cls_num = {0: 1, 18: num_images - 2, 92: 1}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'custom_balanced':
        target_id = start_id
        target_clses = [0, 18, 92]
        cls_num = {0: num_images // 3, 18: num_images // 3, 92: num_images // 3}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'sim':
        target_id = start_id
        target_clses = random.sample(range(num_classes), num_target_cls)
        random_num = split_integer(num_images, num_target_cls)
        cls_num = {target_clses[i]: random_num[i] for i in range(num_target_cls)}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                for _ in range(cls_num[label]):
                    images.append(image.float().to(device))
                    labels.append(torch.as_tensor((label,), device=device))
                    cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'unique':
        target_id = start_id
        uniform_clses = random.sample(range(num_classes), num_images)
        num_per_cls = 1
        cls_cnt = {cls: 0 for cls in uniform_clses}
        while min(list(cls_cnt.values())) < num_per_cls:
            image, label = dataset[target_id]
            if label in uniform_clses and cls_cnt[label] < num_per_cls:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)
    images = torch.stack(images)
    labels = torch.cat(labels)
    return images, labels, target_id


def get_target_data(dataset, labels, start_id=0, device='cpu'):
    images = []
    target_id = start_id
    for i in range(len(labels)):
        while True:
            image, label = dataset[target_id]
            if label == labels[i]:
                images.append(image.float().to(device))
                break
            target_id += 1
            target_id = target_id % len(dataset)
    images = torch.stack(images)
    return images, target_id


def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)),
                                             shuffle=shuffle, drop_last=True)
    return dataloader


def get_log_id(res_dir='logs/Experiment'):
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


def get_grads(outs, labels, model, loss_fn, rec=False):
    loss = loss_fn(outs, labels)
    model.zero_grad()
    grads = torch.autograd.grad(loss, model.parameters()) if rec else torch.autograd.grad(loss, model.fc.parameters())
    grads = list((_.detach().cpu().clone() for _ in grads))
    return grads


def degrade_grads(grads,
                  defense_method='dp',
                  model_name='resnet18',
                  param_value=0.001,
                  images=None,
                  model=None):
    from defense import additive_noise, gradient_clipping, \
        gradient_compression, perturb_representation
    if defense_method == 'dp':
        lossy_grads = additive_noise(grads, param_value)
    elif defense_method == 'clip':
        lossy_grads = gradient_clipping(grads, param_value)
    elif defense_method == 'sparse':
        lossy_grads = gradient_compression(grads, param_value)
    elif defense_method == 'perturb':
        assert model_name == 'resnet18', 'Soteria Perturbation only applicable to resnet18'
        lossy_grads = perturb_representation(grads, model, images, param_value)
    return lossy_grads


def post_process_emb(embedding, model, device, alpha=0.01):
    embedding = embedding.to(device)
    # Feed embedding into FC-Layer to get probabilities.
    out = model.fc(embedding) * alpha
    prob = torch.softmax(out, dim=-1)
    return prob


def collect_cls_variables(embeddings, outs, gt_label, num_classes, model, loss_fn):
    # Sample indexes for each class
    indexes = [np.where(gt_label.cpu().numpy() == i)[0] for i in range(num_classes)]
    cls_embeddings, cls_outs, cls_probs, cls_wgrad, cls_bgrad = {}, {}, {}, {}, {}
    # mods = []
    for i in range(num_classes):
        if len(indexes[i]) > 0:
            # Average embeddings for batch_i (samples of class i)
            cls_embedding = embeddings[indexes[i]]
            # print(cls_embedding)
            cls_emb_mod = torch.norm(cls_embedding, dim=-1)
            # mods.append(cls_emb_mod)
            cls_embeddings[i] = cls_embedding.mean(dim=0)
            # print(cls_embeddings[i])
            # print('*****************')
            # Average outputs for batch_i (samples of class i)
            cls_out = outs[indexes[i]]
            # print(i, cls_embedding.sum(dim=-1), torch.var(cls_embedding.sum(dim=-1), dim=0))
            # print(i, cls_out, torch.var(cls_out, dim=0))
            cls_outs[i] = cls_out.mean(dim=0)
            # Gt labels for batch_i (samples of class i)
            cls_label = gt_label[indexes[i]]
            # Average loss values for batch_i (samples of class i)
            cls_loss = loss_fn(cls_out, cls_label)
            model.zero_grad()
            # Average grads for batch_i (samples of class i)
            cls_grads = torch.autograd.grad(cls_loss, model.fc.parameters(), retain_graph=True)
            # Gradients of last fc layer, for weight and bias
            cls_grads = list((_.detach().cpu().clone() for _ in cls_grads))[-2:]
            cls_wgrad[i], cls_bgrad[i] = cls_grads[0][i], cls_grads[1][i]
            # Average softmax Probabilities for batch_i (samples of class i)
            cls_probs[i] = torch.softmax(cls_out, dim=1).mean(dim=0)
    # mods = torch.cat(mods)
    # print(torch.min(mods), torch.max(mods), torch.var(mods))
    return indexes, cls_embeddings, cls_outs, cls_probs, cls_wgrad, cls_bgrad


def print_util(string, log_file):
    print(string)
    print(string, file=log_file)


def cal_mse(x1, x2):
    return torch.mean((x1.cpu() - x2.cpu()) ** 2)


# Get ground-truth labels and its instance statistic
def get_label_stats(gt_label, num_classes):
    LabelCounter = dict(Counter(gt_label.cpu().numpy()))
    labels = list(sorted(LabelCounter.keys()))
    existences = [1 if i in labels else 0 for i in range(num_classes)]
    num_instances = [LabelCounter[i] if i in labels else 0 for i in range(num_classes)]
    num_instances_nonzero = [item[1] for item in sorted(LabelCounter.items(), key=lambda x: x[0])]
    return labels, existences, num_instances, num_instances_nonzero


def get_irlg_res(cls_rec_probs, b_grad, gt_label, num_classes, num_images, log_file, simplified=False):
    labels, existences, num_instances, num_instances_nonzero = get_label_stats(gt_label, num_classes)
    # Recovered Labels
    rec_instances, mod_rec_instances = sim_iLRG(cls_rec_probs, b_grad, labels, num_images) if simplified else iLRG(
        cls_rec_probs,
        b_grad,
        num_classes,
        num_images)
    rec_labels = labels if simplified else list(np.where(rec_instances > 0)[0])
    rec_instances_nonzero = rec_instances if simplified else rec_instances[rec_labels]
    rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
    # Calculate Class-wise Acc, Instance-wise Acc and Recall
    leacc = 1.0 if simplified else accuracy_score(existences, rec_existences)
    lnacc = accuracy_score(num_instances_nonzero if simplified else num_instances, list(rec_instances))
    irec = sum([rec_instances[i] if rec_instances[i] <= num_instances_nonzero[i] else num_instances_nonzero[i] for i in
                range(len(labels))]) / num_images if simplified else sum(
        [rec_instances[i] if rec_instances[i] <= num_instances[i] else num_instances[i] for i in labels]) / num_images
    # Print results
    print_util('Ground-truth Labels: ' + ','.join(str(l) for l in labels), log_file=log_file)
    print_util('Ground-truth Num of Instances: ' + ','.join(str(num_instances[l]) for l in labels), log_file=log_file)
    print_util('Our Recovered Labels: ' + ','.join(str(l) for l in rec_labels) + ' | LeAcc: %.3f' % leacc,
               log_file=log_file)
    prefix = 'Our Recovered Num of Instances by Simplified Method: ' if simplified else 'Our Recovered Num of Instances: '
    print_util(prefix + ','.join(str(l) for l in list(rec_instances_nonzero)) +
               ' | LnAcc: %.3f | IRec: %.3f' % (
                   lnacc, irec),
               log_file=log_file)
    res = [rec_labels, rec_instances_nonzero, rec_instances, existences, mod_rec_instances]
    metrics = [leacc, lnacc, irec]
    return res, metrics


def get_other_res(w_grad, num_classes, existences, log_file, attack_method='idlg', num_images=None):
    if attack_method == 'idlg':
        rec_labels = list(idlg_infer(w_grad.transpose(0, 1)))
        rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
        leacc = accuracy_score(existences, rec_existences)
        print_util('iDLG Recovered Labels: ' + ','.join(str(l) for l in rec_labels) + ' | LeAcc: %.3f' % leacc,
                   log_file=log_file)
    elif attack_method == 'gi':
        rec_labels = list(gi_infer(w_grad.transpose(0, 1)))
        rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
        leacc = accuracy_score(existences, rec_existences)
        print_util('GradInversion Recovered Labels: ' + ','.join(str(l) for l in rec_labels) + ' | LeAcc: %.3f' % leacc,
                   log_file=log_file)
    elif attack_method == 'svd':
        rec_labels = list(svd_infer(w_grad.transpose(0, 1), num_classes=num_classes, gt_k=num_images, epsilon=1e-10))
        rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
        leacc = accuracy_score(existences, rec_existences)
        print_util('SVD Recovered Labels: ' + ','.join(str(l) for l in rec_labels) + ' | LeAcc: %.3f' % leacc,
                   log_file=log_file)
    else:
        raise NotImplementedError('Attack method not implemented.')
    return leacc
