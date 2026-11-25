"""Restore a batch's labels with iLRG or a baseline, score them against the true labels and log them."""

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

from ..utils import print_util
from .baselines import gi_infer, idlg_infer, svd_infer
from .ilrg import iLRG, sim_iLRG

# Name of each comparison attack in the log
OTHER_ATTACK_NAMES = {'idlg': 'iDLG', 'gi': 'GradInversion', 'svd': 'SVD'}

SVD_EPSILON = 1e-10


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
