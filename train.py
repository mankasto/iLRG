"""Part of the code from https://github.com/JonasGeiping/invertinggradients"""

import os
import math
import torch
import numpy as np
from options import options
from models import get_model
from loss import Classification
from datasets import get_dataset
from collections import defaultdict
from scheduler import GradualWarmupScheduler
from utils import Config, get_device, get_dataloader

NON_BLOCKING = False


def set_optimizer(model, defs):
    """Build model optimizer and scheduler from defs.

    The linear scheduler drops the learning rate in intervals.
    # Example: epochs=160 leads to drops at 60, 100, 140.
    """
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'linear':
        milestone = defs.iterations if defs.iter_train else defs.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[milestone // 2.667, milestone // 1.6,
                                                                     milestone // 1.142], gamma=0.1)

    if defs.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)

    return optimizer, scheduler


def step(model, loss_fn, dataloader, optimizer, scheduler, defs, setup, stats):
    """Step through one epoch."""
    epoch_loss, epoch_metric = 0, 0
    for batch, (inputs, targets) in enumerate(dataloader):
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**setup)
        targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)

        # Get loss
        outputs, _ = model(inputs)
        loss, _, _ = loss_fn(outputs, targets)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

        if defs.scheduler == 'cyclic':
            scheduler.step()
    if defs.scheduler == 'linear':
        scheduler.step()

    stats['train_losses'].append(epoch_loss / (batch + 1))
    stats['train_' + name].append(epoch_metric / (batch + 1))


def iter_step(model, loss_fn, inputs, targets, optimizer, scheduler, setup, stats):
    """Step through one iteration."""
    # Prep Mini-Batch
    optimizer.zero_grad()
    # Transfer to GPU
    inputs = inputs.to(**setup)
    targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
    # Get loss
    outputs, _ = model(inputs)
    loss, _, _ = loss_fn(outputs, targets)
    iter_loss = loss.item()
    loss.backward()
    optimizer.step()
    metric, name, _ = loss_fn.metric(outputs, targets)
    iter_metric = metric.item()
    scheduler.step()
    stats['train_losses'].append(iter_loss)
    stats['train_' + name].append(iter_metric)


def print_status(loss_fn, optimizer, defs, stats, epoch=0, iteration=0):
    """Print basic console printout every defs.validation epochs."""
    current_lr = optimizer.param_groups[0]['lr']
    name, pt_format = loss_fn.metric()
    if defs.iter_train:
        print(f'Iteration: {iteration + 1}| lr: {current_lr:.4f} | '
              f'Train loss is {stats["train_losses"][-1]:6.4f}, Train {name}: {stats["train_" + name][-1]:{pt_format}} | '
              f'Val loss is {stats["valid_losses"][-1]:6.4f}, Val {name}: {stats["valid_" + name][-1]:{pt_format}} |')
    else:
        print(f'Epoch: {epoch + 1}| lr: {current_lr:.4f} | '
              f'Train loss is {stats["train_losses"][-1]:6.4f}, Train {name}: {stats["train_" + name][-1]:{pt_format}} | '
              f'Val loss is {stats["valid_losses"][-1]:6.4f}, Val {name}: {stats["valid_" + name][-1]:{pt_format}} |')


def validate(model, loss_fn, dataloader, defs, setup, stats):
    """Validate model effectiveness of val dataset."""
    epoch_loss, epoch_metric = 0, 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(dataloader):
            # Transfer to GPU
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)

            # Get loss and metric
            outputs, _ = model(inputs)
            loss, _, _ = loss_fn(outputs, targets)
            metric, name, _ = loss_fn.metric(outputs, targets)

            epoch_loss += loss.item()
            epoch_metric += metric.item()

            if defs.dryrun:
                break

    stats['valid_losses'].append(epoch_loss / (batch + 1))
    stats['valid_' + name].append(epoch_metric / (batch + 1))


def train(model, loss_fn, trainloader, validloader, defs, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, defs)

    for epoch in range(defs.epochs):
        model.train()
        step(model, loss_fn, trainloader, optimizer, scheduler, defs, setup, stats)

        if (epoch + 1) % defs.epoch_interval == 0 or epoch == 0:
            model.eval()
            validate(model, loss_fn, validloader, defs, setup, stats)
            # Print information about loss and accuracy
            print_status(loss_fn, optimizer, defs, stats, epoch)
            if defs.mid_save and 0 < epoch < defs.epochs - 1:
                file = f'{defs.model}_{defs.dataset}_Epoch{epoch + 1}.pth'
                torch.save(model.state_dict(), os.path.join(defs.save_dir, file))

        if defs.dryrun:
            break
        if not (np.isfinite(stats['train_losses'][-1])):
            print('Loss is NaN/Inf ... terminating early ...')
            break
    # Final validation and saving model
    validate(model, loss_fn, validloader, defs, setup, stats)
    # Print information about loss and accuracy
    print_status(loss_fn, optimizer, defs, stats, epoch)
    file = f'{defs.model}_{defs.dataset}_Epoch{defs.epochs}.pth'
    torch.save(model.state_dict(), os.path.join(defs.save_dir, file))

    return stats


def iter_train(model, loss_fn, trainloader, validloader, defs,
               setup=dict(dtype=torch.float, device=torch.device('cpu'))):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, defs)
    epoch_iter = len(trainloader) / defs.batch_size
    epochs = math.ceil(defs.iterations / epoch_iter)
    iter_cnt = 0
    for epoch in range(epochs):
        model.train()
        for batch, (inputs, targets) in enumerate(trainloader):
            iter_step(model, loss_fn, inputs, targets, optimizer, scheduler, setup, stats)
            if (iter_cnt + 1) % defs.iter_interval == 0 or iter_cnt == 0:
                model.eval()
                validate(model, loss_fn, validloader, defs, setup, stats)
                # Print information about loss and accuracy
                print_status(loss_fn, optimizer, defs, stats, iteration=iter_cnt)
                if defs.mid_save and 0 < iter_cnt < defs.iterations - 1:
                    file = f'{defs.model}_{defs.dataset}_Iter{iter_cnt + 1}.pth'
                    torch.save(model.state_dict(), os.path.join(defs.save_dir, file))
            iter_cnt += 1
            if iter_cnt >= defs.iterations:
                break
        if iter_cnt >= defs.iterations:
            break
        if defs.dryrun:
            break
        if not (np.isfinite(stats['train_losses'][-1])):
            print('Loss is NaN/Inf ... terminating early ...')
            break
    file = f'{defs.model}_{defs.dataset}_Iter{defs.iterations}.pth'
    torch.save(model.state_dict(), os.path.join(defs.save_dir, file))

    return stats


if __name__ == '__main__':
    args = options().parse_args()
    defs = Config({'iter_train': args.iter_train,
                   'epochs': args.epochs,
                   'iterations': args.iters,
                   'batch_size': args.batch_size,
                   'optimizer': args.optimizer,
                   'lr': args.lr,
                   'scheduler': args.scheduler,
                   'weight_decay': args.weight_decay,
                   'warmup': args.warmup,
                   'epoch_interval': args.epoch_interval,
                   'iter_interval': args.iter_interval,
                   'dryrun': args.dryrun,
                   'model': args.model,
                   'dataset': args.dataset,
                   'mid_save': args.mid_save,
                   'save_dir': args.model_path})
    device = get_device(use_cuda=False if args.cpu else True)
    setup = dict(device=device, dtype=torch.float)

    dataset_params = get_dataset(dataset=args.dataset,
                                 data_path=args.data_path,
                                 model=args.model)
    img_shape, num_classes, channel, hidden, dataset = dataset_params
    train_data, valid_data = dataset
    train_loader = get_dataloader(train_data, batch_size=defs.batch_size, shuffle=True)
    valid_loader = get_dataloader(valid_data, batch_size=defs.batch_size, shuffle=False)
    model = get_model(model_name=args.model,
                      net_params=(num_classes, channel, hidden),
                      device=device,
                      batchnorm=args.batchnorm,
                      dropout=args.dropout,
                      silu=args.silu)
    criterion = Classification()
    if args.iter_train:
        stats = iter_train(model, criterion, train_loader, valid_loader, defs, setup=setup)
    else:
        stats = train(model, criterion, train_loader, valid_loader, defs, setup=setup)
    stats = defaultdict(list)
    print(stats)
