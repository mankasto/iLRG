import os
import re
import torch
import random
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from .consts import CUSTOM_DISTRIBUTION_CLASSES
from .methods import iLRG, sim_iLRG, idlg_infer, gi_infer, svd_infer

# Name of each comparison attack in the log
OTHER_ATTACK_NAMES = {'idlg': 'iDLG', 'gi': 'GradInversion', 'svd': 'SVD'}
SVD_EPSILON = 1e-10
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
    """Draw a batch whose labels follow `data_distribution`.

    'random' samples indices uniformly; the other distributions fix how many images to take
    from each class and scan the dataset from `start_id`. Returns the images, the labels and
    the index the next batch starts from.
    """
    if data_distribution == 'random':
        indices = random.sample(range(len(dataset)), num_images)
        samples = [dataset[index] for index in indices]
        next_id = indices[0]
    else:
        quotas, batch_size = _class_quotas(data_distribution, num_images, num_classes, num_uniform_cls,
                                           num_target_cls)
        samples, next_id = _scan_for_classes(dataset, start_id, quotas, batch_size,
                                             repeat_first=data_distribution == 'sim')
    images = torch.stack([image.float().to(device) for image, _ in samples])
    labels = torch.cat([torch.as_tensor((label,), device=device) for _, label in samples])
    return images, labels, next_id


def _class_quotas(distribution, num_images, num_classes, num_uniform_cls, num_target_cls):
    """Images to take per class for a scanned distribution, and the batch size the scan stops at.

    extreme: all images from one random class. balanced: the same number from num_uniform_cls
    random classes. random2 and sim: random counts summing to num_images over num_target_cls
    random classes. custom_*: the classes in CUSTOM_DISTRIBUTION_CLASSES. unique: one image from
    each of num_images random classes. Unknown names give no quotas, so an empty batch.
    """
    if distribution == 'extreme':
        return {random.randint(0, num_classes - 1): num_images}, num_images
    if distribution == 'balanced':
        classes = random.sample(range(num_classes), num_uniform_cls)
        per_class = num_images // num_uniform_cls
        return {label: per_class for label in classes}, per_class * num_uniform_cls
    if distribution in ('random2', 'sim'):
        classes = random.sample(range(num_classes), num_target_cls)
        return dict(zip(classes, split_integer(num_images, num_target_cls))), num_images
    if distribution == 'custom_imbalanced':
        first, middle, last = CUSTOM_DISTRIBUTION_CLASSES
        return {first: 1, middle: num_images - 2, last: 1}, num_images
    if distribution == 'custom_balanced':
        per_class = num_images // len(CUSTOM_DISTRIBUTION_CLASSES)
        return {label: per_class for label in CUSTOM_DISTRIBUTION_CLASSES}, num_images
    if distribution == 'unique':
        return {label: 1 for label in random.sample(range(num_classes), num_images)}, num_images
    return {}, 0


def _scan_for_classes(dataset, start_id, quotas, batch_size, repeat_first=False):
    """Walk the dataset from start_id (wrapping around) until batch_size samples are kept.

    A sample is kept while its class is below its quota. With repeat_first, the first sample
    found for a class fills that class's whole quota. Returns the samples and the next index.
    """
    samples = []
    counts = {label: 0 for label in quotas}
    index = start_id
    while sum(counts.values()) < batch_size:
        image, label = dataset[index]
        if label in quotas and counts[label] < quotas[label]:
            copies = quotas[label] if repeat_first else 1
            samples.extend([(image, label)] * copies)
            counts[label] += copies
        index = (index + 1) % len(dataset)
    return samples, index


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
    from .defense import additive_noise, gradient_clipping, \
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
    for i in range(num_classes):
        if len(indexes[i]) > 0:
            # Average embeddings for batch_i (samples of class i)
            cls_embedding = embeddings[indexes[i]]
            cls_embeddings[i] = cls_embedding.mean(dim=0)
            # Average outputs for batch_i (samples of class i)
            cls_out = outs[indexes[i]]
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
    return indexes, cls_embeddings, cls_outs, cls_probs, cls_wgrad, cls_bgrad


def print_util(string, log_file):
    print(string)
    print(string, file=log_file)


def get_label_stats(gt_label, num_classes):
    """Labels present in the batch, per-class existences, per-class counts and counts of present labels."""
    LabelCounter = dict(Counter(gt_label.cpu().numpy()))
    labels = list(sorted(LabelCounter.keys()))
    existences = [1 if i in labels else 0 for i in range(num_classes)]
    num_instances = [LabelCounter[i] if i in labels else 0 for i in range(num_classes)]
    num_instances_nonzero = [item[1] for item in sorted(LabelCounter.items(), key=lambda x: x[0])]
    return labels, existences, num_instances, num_instances_nonzero


def get_irlg_res(cls_rec_probs, b_grad, gt_label, num_classes, num_images, log_file, simplified=False):
    """Restore the batch labels with iLRG (or, simplified, with the true classes known) and log them.

    Returns ([restored labels, their counts, counts per class, true existences, counts adjusted to
    the batch size], [label existence accuracy, label number accuracy, instance recall]).
    """
    labels, existences, num_instances, num_instances_nonzero = get_label_stats(gt_label, num_classes)
    if simplified:
        rec_instances, mod_rec_instances = sim_iLRG(cls_rec_probs, b_grad, labels, num_images)
        rec_labels = labels
        rec_instances_nonzero = rec_instances
        true_counts = num_instances_nonzero
        irec_pairs = [(rec_instances[i], num_instances_nonzero[i]) for i in range(len(labels))]
    else:
        rec_instances, mod_rec_instances = iLRG(cls_rec_probs, b_grad, num_classes, num_images)
        rec_labels = list(np.where(rec_instances > 0)[0])
        rec_instances_nonzero = rec_instances[rec_labels]
        true_counts = num_instances
        irec_pairs = [(rec_instances[i], num_instances[i]) for i in labels]
    rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
    # Class-wise accuracy, instance-wise accuracy and instance recall
    leacc = 1.0 if simplified else accuracy_score(existences, rec_existences)
    lnacc = accuracy_score(true_counts, list(rec_instances))
    irec = sum([rec if rec <= true else true for rec, true in irec_pairs]) / num_images
    print_util('Ground-truth Labels: ' + ','.join(str(l) for l in labels), log_file=log_file)
    print_util('Ground-truth Num of Instances: ' + ','.join(str(num_instances[l]) for l in labels), log_file=log_file)
    print_util('Our Recovered Labels: ' + ','.join(str(l) for l in rec_labels) + ' | LeAcc: %.3f' % leacc,
               log_file=log_file)
    prefix = 'Our Recovered Num of Instances by Simplified Method: ' if simplified else 'Our Recovered Num of Instances: '
    print_util(prefix + ','.join(str(l) for l in list(rec_instances_nonzero)) +
               ' | LnAcc: %.3f | IRec: %.3f' % (lnacc, irec),
               log_file=log_file)
    res = [rec_labels, rec_instances_nonzero, rec_instances, existences, mod_rec_instances]
    metrics = [leacc, lnacc, irec]
    return res, metrics


def get_other_res(w_grad, num_classes, existences, log_file, attack_method='idlg', num_images=None):
    """Restore the batch labels with iDLG, GradInversion or the SVD method; log them and return LeAcc."""
    w_grad_transposed = w_grad.transpose(0, 1)
    if attack_method == 'idlg':
        rec_labels = idlg_infer(w_grad_transposed)
    elif attack_method == 'gi':
        rec_labels = gi_infer(w_grad_transposed)
    elif attack_method == 'svd':
        rec_labels = svd_infer(w_grad_transposed, num_classes=num_classes, gt_k=num_images, epsilon=SVD_EPSILON)
    else:
        raise NotImplementedError('Attack method not implemented.')
    rec_labels = list(rec_labels)
    rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
    leacc = accuracy_score(existences, rec_existences)
    print_util(OTHER_ATTACK_NAMES[attack_method] + ' Recovered Labels: ' + ','.join(str(l) for l in rec_labels)
               + ' | LeAcc: %.3f' % leacc, log_file=log_file)
    return leacc
