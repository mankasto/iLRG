"""Repeatable code parts concerning data loading.
Modified From https://github.com/JonasGeiping/invertinggradients"""

import os
import torch
import consts
import torchvision
import torchvision.transforms as transforms


def get_dataset(dataset, aug=True, data_path='data',
                normalize=True, split='all', model='lenet5'):
    path = os.path.expanduser(data_path)
    if dataset == 'CIFAR10':
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        if model == 'lenet5':
            hidden = 400
        elif model == 'lenetzhu':
            hidden = 768
        elif 'vgg' in model:
            hidden = 512
        else:
            hidden = None
        trainset, validset = _build_cifar10(path, aug, normalize)
    elif dataset == 'CIFAR100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        if model == 'lenet5':
            hidden = 400
        elif model == 'lenetzhu':
            hidden = 768
        elif 'vgg' in model:
            hidden = 512
        else:
            hidden = None
        trainset, validset = _build_cifar100(path, aug, normalize)
    elif dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 3
        if model == 'lenet5':
            hidden = 256
        elif model == 'lenetzhu':
            hidden = 588
        else:
            hidden = None
        trainset, validset = _build_mnist(path, aug, normalize)
    elif dataset == 'MNIST_GRAY':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        if model == 'lenet5':
            hidden = 256
        elif model == 'lenetzhu':
            hidden = 588
        else:
            hidden = None
        trainset, validset = _build_mnist_gray(path, aug, normalize)
    elif dataset == 'ImageNet':
        shape_img = (224, 224)
        num_classes = 1000
        channel = 3
        if 'vgg' in model:
            hidden = 7 * 7 * 512
        else:
            hidden = None
        trainset, validset = _build_imagenet(path, aug, normalize)
    else:
        exit('unknown data')
    if split == 'train':
        dst = trainset
    elif split == 'val':
        dst = validset
    else:
        dst = (trainset, validset)
    return shape_img, num_classes, channel, hidden, dst


def _build_cifar10(data_path, augmentations=True, normalize=True):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if consts.cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = consts.cifar10_mean, consts.cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True,
                                             transform=transforms.ToTensor())

    if consts.cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = consts.cifar100_mean, consts.cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if consts.mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = consts.mnist_mean, consts.mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if consts.mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = consts.mnist_mean, consts.mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_imagenet(data_path, augmentations=True, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    if consts.imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = consts.imagenet_mean, consts.imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _get_meanstd(trainset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std
