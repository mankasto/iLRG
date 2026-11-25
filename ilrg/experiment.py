"""Label restoration experiments behind ``python -m ilrg``.

Each trial samples a batch, computes its gradients and restores the batch labels with iLRG;
options add other label attacks for comparison, an error analysis and image reconstruction.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import TextIO

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset

from .federated import data
from .federated.data import get_data, get_dataset
from .federated.defense import params
from .federated.gradients import degrade_grads, get_grads
from .federated.models import get_model
from .federated.train import train_model, training_config
from .options import options
from .reconstruction.reconstructor import GradientReconstructor
from .restoration.analysis import ErrorAnalysis, collect_cls_variables
from .restoration.ilrg import get_emb, post_process_emb
from .restoration.results import get_irlg_res, get_other_res
from .utils import AverageMeter, get_device, get_log_file, get_params_string, print_util, set_random_seed

TRIAL_SEPARATOR = '*' * 63
SUMMARY_SEPARATOR = '-' * 63
OTHER_ATTACKS = ('idlg', 'gi', 'svd')
IG_DEFAULT_LR = 0.1
DLG_DEFAULT_LR = 1.0


@dataclass
class Run:
    """Settings, model and data that stay fixed over the trials of a run."""
    args: argparse.Namespace
    log_file: TextIO
    model: nn.Module
    criterion: nn.Module
    device: str
    target_data: Dataset  # the split batches are drawn from
    valid_data: Dataset  # also the auxiliary data of --init auxiliary
    image_shape: tuple  # (channels, height, width)
    mean_std: tuple  # dataset mean and std as (C, 1, 1) tensors; None without --rec_img

    @property
    def setup(self):
        return dict(device=self.device, dtype=torch.float)

    def log(self, text):
        print_util(text, log_file=self.log_file)


def start_run(args):
    """Set the output folders, seed every RNG, open the run log and log the settings."""
    args.log_dir = os.path.join('results', 'runs', args.exp_name)
    args.image_dir = os.path.join('results', 'reconstructions', args.exp_name)
    set_random_seed(args.seed)
    log_file = get_log_file(args.log_dir)
    current_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
    print_util('running experiments at {}'.format(current_time), log_file=log_file)
    print_util(get_params_string(vars(args)), log_file=log_file)
    return log_file


def load_or_train_model(args, model, train_data, valid_data, device, log_file):
    """Load the checkpoint the settings name, or train the model (which saves one)."""
    if args.iter_train:
        file = f'{args.model}_{args.dataset}_Iter{args.iters}.pth'
    else:
        file = f'{args.model}_{args.dataset}_Epoch{args.epochs}.pth'
    try:
        model.load_state_dict(torch.load(os.path.join(args.model_path, file), map_location=device))
        print_util(f'Model loaded from file {file}.', log_file=log_file)
    except FileNotFoundError:
        print_util('Training the model ...', log_file=log_file)
        train_model(model, training_config(args), train_data, valid_data, setup=dict(device=device, dtype=torch.float))


def image_mean_std(dataset, device):
    """Dataset mean and std as (C, 1, 1) tensors, to undo the input normalisation."""
    name = 'mnist' if dataset.lower() == 'mnist_gray' else dataset.lower()
    mean = torch.as_tensor(getattr(data, f'{name}_mean'))[:, None, None].to(device)
    std = torch.as_tensor(getattr(data, f'{name}_std'))[:, None, None].to(device)
    return mean, std


def reconstruction_config(args):
    """Settings of the gradient-inversion attack chosen by --optim ('ig' or 'dlg')."""
    if args.optim == 'ig':
        return dict(signed=args.signed,
                    boxed=True,
                    cost_fn=args.cost_fn,
                    indices=args.indices,
                    weights=args.weights,
                    lr=args.rec_lr if args.rec_lr is not None else IG_DEFAULT_LR,
                    optim='adam',
                    restarts=args.restarts,
                    max_iterations=args.max_iterations,
                    total_variation=args.tv,
                    l2_norm=args.l2,
                    init=args.init,
                    filter='median' if args.dataset == 'ImageNet' else 'none',
                    lr_decay=True,
                    scoring_choice=args.scoring_choice)
    if args.optim == 'dlg':
        return dict(signed=False,
                    boxed=False,
                    cost_fn='l2',
                    indices='def',
                    weights='equal',
                    lr=args.rec_lr if args.rec_lr is not None else DLG_DEFAULT_LR,
                    optim='LBFGS',
                    restarts=args.restarts,
                    max_iterations=args.max_iterations,
                    total_variation=args.tv,
                    l2_norm=args.l2,
                    init=args.init,
                    filter='none',
                    lr_decay=False,
                    scoring_choice=args.scoring_choice)
    raise ValueError("Unknown --optim '{}', expected 'ig' or 'dlg'.".format(args.optim))


def defense_parameter(args):
    """Value of the chosen defense's setting, looked up by the name defense.params gives.

    Kept as is: defense.params names 'prune_rate' for 'perturb' while the setting here is
    'prune_ratio', so the perturb defense stops with a NameError.
    """
    settings = {'noise_std': args.noise_std,
                'clip_bound': args.clip_bound,
                'sparse_ratio': args.sparse_ratio,
                'prune_ratio': args.prune_ratio}
    return eval(params[args.defense_method], settings)


def forward_in_chunks(run, gt_data, gt_label):
    """Forward a large ImageNet batch in chunks of --max_size and average the chunk gradients."""
    args = run.args
    assert args.num_images % args.max_size == 0, 'Number of images should be an integral multiple of max size'
    split_data = torch.split(gt_data, args.max_size, 0)
    split_label = torch.split(gt_label, args.max_size, 0)
    grads, embeddings, outs = [], [], []
    for i in range(args.num_images // args.max_size):
        split_outs, split_embeddings = run.model(split_data[i])
        grads.append(get_grads(outs=split_outs, labels=split_label[i], model=run.model, loss_fn=run.criterion,
                               rec=args.rec_img))
        outs.append(split_outs)
        embeddings.append(split_embeddings)
        print('Finish splitting {}!'.format(i))
    grads = np.mean(np.stack(np.array(grads)), axis=0)
    embeddings = np.stack(np.array(embeddings))
    outs = np.stack(np.array(outs))
    return outs, embeddings, grads


def forward_batch(run, gt_data, gt_label):
    """Model outputs and batch-averaged gradients, plus per-class averages for --analysis."""
    args = run.args
    chunked = args.dataset == 'ImageNet' and args.num_images > args.max_size
    if chunked:
        outs, embeddings, grads = forward_in_chunks(run, gt_data, gt_label)
    else:
        outs, embeddings = run.model(gt_data)
    class_stats = None
    if args.analysis:
        class_stats = collect_cls_variables(embeddings=embeddings,
                                            outs=outs,
                                            gt_label=gt_label,
                                            num_classes=args.num_classes,
                                            model=run.model,
                                            loss_fn=run.criterion)
    if not chunked:
        grads = get_grads(outs=outs, labels=gt_label, model=run.model, loss_fn=run.criterion, rec=args.rec_img)
        print('Finish getting grads!')
    return outs, grads, class_stats


def log_batch_accuracy(run, outs, gt_label):
    num_images = run.args.num_images
    probs = torch.softmax(outs, dim=-1)
    preds = torch.max(probs, 1)[1].cpu()
    correct = (preds == gt_label.cpu()).sum().item()
    run.log('%d/%d, Acc for this batch: %.3f' % (correct, num_images, correct / num_images))


def restore_class_probabilities(run, w_grad, b_grad, class_stats, analysis):
    """Restore each class's average embedding from its gradient row and map it to probabilities.

    With --analysis, each class is also compared with its true averages.
    """
    rec_probs = []
    for i in range(run.args.num_classes):
        rec_emb = get_emb(w_grad[i], b_grad[i])
        rec_prob = post_process_emb(embedding=rec_emb, model=run.model, device=run.device, alpha=run.args.alpha)
        rec_probs.append(rec_prob)
        if analysis is not None:
            analysis.add_class(i, rec_emb, rec_prob, w_grad[i], b_grad[i], class_stats)
    return rec_probs


def restore_labels(run, rec_probs, b_grad, gt_label, meters, custom_counts):
    """Restore the batch labels with iLRG, log them and record the accuracies.

    Returns the class existences of the batch and iLRG's counts adjusted to the batch size.
    """
    args = run.args
    if args.estimate:
        rec_probs = [torch.ones(args.num_classes) / args.num_classes for _ in range(args.num_classes)]
    res, metrics = get_irlg_res(cls_rec_probs=rec_probs,
                                b_grad=b_grad,
                                gt_label=gt_label,
                                num_classes=args.num_classes,
                                num_images=args.num_images,
                                log_file=run.log_file,
                                simplified=args.simplified)
    _, _, rec_instances, existences, mod_rec_instances = res
    for label, meter in custom_counts.items():
        meter.update(rec_instances[label])
    for name, value in zip(('ilrg_leacc', 'ilrg_lnacc', 'ilrg_irec'), metrics):
        meters[name].update(value)
    return existences, mod_rec_instances


def compare_other_attacks(run, w_grad, existences, meters):
    """Restore the labels with iDLG, GradInversion and the SVD method, and record their accuracies."""
    for attack in OTHER_ATTACKS:
        accuracy = get_other_res(w_grad=w_grad,
                                 num_classes=run.args.num_classes,
                                 existences=existences,
                                 log_file=run.log_file,
                                 attack_method=attack,
                                 num_images=run.args.num_images)
        meters[attack + '_leacc'].update(accuracy)


def reconstruction_labels(run, gt_label, mod_rec_instances):
    """Labels handed to the reconstruction: None (optimised with the images), the true ones or iLRG's."""
    args = run.args
    if not args.fix_labels:
        return None
    if args.gt_labels:
        return gt_label
    labels = []
    for label in range(args.num_classes):
        if mod_rec_instances[label] > 0:
            labels.extend([label] * mod_rec_instances[label])
    return torch.from_numpy(np.array(labels)).long().to(run.device)


def save_images(images, num_images, image_dir, prefix):
    for j in range(num_images):
        torchvision.utils.save_image(images[j:j + 1, ...], os.path.join(image_dir, f'{prefix}_{j}.png'))


def reconstruct_images(run, grads, gt_data, gt_label, mod_rec_instances):
    """Invert the gradients into images; save the ground truth and, with --save_image, the result."""
    args, model = run.args, run.model
    model.eval()
    grads = [g.to(**run.setup) for g in grads]
    rec_machine = GradientReconstructor(model=model,
                                                 config=reconstruction_config(args),
                                                 mean_std=run.mean_std,
                                                 num_images=args.num_images,
                                                 loss_thresh=args.loss_thresh,
                                                 rec_exp_dir=args.image_dir)
    labels = reconstruction_labels(run, gt_label, mod_rec_instances)
    print(gt_label, labels)
    run.log(' ')
    run.log('Start reconstructing images')
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)
    mean, std = run.mean_std
    save_images(torch.clamp(gt_data * std + mean, 0, 1), args.num_images, args.image_dir, 'gt')
    output, _ = rec_machine.reconstruct(grads, labels,
                                        img_shape=run.image_shape,
                                        dryrun=args.dryrun,
                                        aux_data=run.valid_data)
    run.log('End reconstructing images')
    if args.save_image and not args.dryrun:
        save_images(torch.clamp(output * std + mean, 0, 1), args.num_images, args.image_dir, 'rec')


def run_trial(run, exp_id, target_id, meters, custom_counts, analysis):
    """One experiment on a fresh batch; returns the dataset index the next batch starts from."""
    args = run.args
    run.log('Start Experiment {}'.format(exp_id + 1))
    target_id = target_id % len(run.target_data)
    run.log('start_id: ' + str(target_id))
    gt_data, gt_label, target_id = get_data(dataset=run.target_data,
                                            num_images=args.num_images,
                                            num_classes=args.num_classes,
                                            data_distribution=args.distribution,
                                            start_id=target_id,
                                            num_uniform_cls=args.num_uniform_cls,
                                            num_target_cls=args.num_target_cls,
                                            device=run.device)
    print('Finish getting data!')
    outs, grads, class_stats = forward_batch(run, gt_data, gt_label)
    print('Finish forward process!')
    log_batch_accuracy(run, outs, gt_label)

    w_grad, b_grad = grads[-2], grads[-1]
    if args.defense:
        # Lossy transformation of the gradients against the attack
        w_grad, b_grad = degrade_grads(grads=[w_grad, b_grad],
                                       defense_method=args.defense_method,
                                       param_value=defense_parameter(args),
                                       model_name=args.model,
                                       images=gt_data,
                                       model=run.model)
    if analysis is not None:
        analysis.start_batch()
    rec_probs = restore_class_probabilities(run, w_grad, b_grad, class_stats, analysis)
    run.log(TRIAL_SEPARATOR)
    if analysis is not None:
        analysis.end_batch()

    existences, mod_rec_instances = restore_labels(run, rec_probs, b_grad, gt_label, meters, custom_counts)
    if args.compare:
        compare_other_attacks(run, w_grad, existences, meters)
    if args.rec_img:
        reconstruct_images(run, grads, gt_data, gt_label, mod_rec_instances)
    return target_id


def log_summary(run, meters, custom_counts, analysis):
    """Mean results over all trials."""
    run.log(SUMMARY_SEPARATOR)
    summary = 'Mean Ours LeAcc: %.3f | Mean Ours LnAcc: %.3f | Mean Ours IRec: %.3f' % (
        meters['ilrg_leacc'].avg, meters['ilrg_lnacc'].avg, meters['ilrg_irec'].avg)
    if run.args.compare:
        summary = 'Mean iDLG LeAcc: %.3f | Mean GI LeAcc: %.3f | Mean SVD LeAcc: %.3f ' % (
            meters['idlg_leacc'].avg, meters['gi_leacc'].avg, meters['svd_leacc'].avg) + summary
    run.log(summary)
    if analysis is not None:
        analysis.print_summary()
    if custom_counts:
        run.log('Mean Rec Instances: ' + ', '.join('Class %d-%d' % (label, meter.avg)
                                                   for label, meter in custom_counts.items()))


def run_experiments(args):
    """Run --num_tries trials with one model and log their mean results."""
    log_file = start_run(args)
    img_shape, num_classes, channel, hidden, (train_data, valid_data) = get_dataset(dataset=args.dataset,
                                                                                   data_path=args.data_path,
                                                                                   model=args.model,
                                                                                   aug=False)
    target_data = train_data if args.split == 'train' else valid_data
    args.num_classes = num_classes
    device = get_device(use_cuda=not args.cpu)

    model = get_model(model_name=args.model,
                      net_params=(num_classes, channel, hidden),
                      device=device,
                      n_hidden=args.n_hidden,
                      n_dim=args.n_dim,
                      batchnorm=args.batchnorm,
                      dropout=args.dropout,
                      silu=args.silu,
                      leaky_relu=args.leaky_relu)
    if args.trained_model:
        load_or_train_model(args, model, train_data, valid_data, device, log_file)
    model.eval()

    run = Run(args=args,
              log_file=log_file,
              model=model,
              criterion=nn.CrossEntropyLoss(),
              device=device,
              target_data=target_data,
              valid_data=valid_data,
              image_shape=(channel, *img_shape),
              mean_std=image_mean_std(args.dataset, device) if args.rec_img else None)
    meters = {name: AverageMeter() for name in ('ilrg_leacc', 'ilrg_lnacc', 'ilrg_irec')}
    meters.update({attack + '_leacc': AverageMeter() for attack in OTHER_ATTACKS})
    custom_counts = {}
    if 'custom' in args.distribution:
        custom_counts = {label: AverageMeter() for label in data.CUSTOM_DISTRIBUTION_CLASSES}
    analysis = ErrorAnalysis(model, device, args.num_images, args.ratio, log_file) if args.analysis else None

    target_id = args.start_id
    for exp_id in range(args.num_tries):
        target_id = run_trial(run, exp_id, target_id, meters, custom_counts, analysis)
    log_summary(run, meters, custom_counts, analysis)


if __name__ == '__main__':
    run_experiments(options().parse_args())
