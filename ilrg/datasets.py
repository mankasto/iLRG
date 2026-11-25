"""Dataset loading.
Modified From https://github.com/JonasGeiping/invertinggradients"""

import os
from typing import NamedTuple

import torchvision
import torchvision.transforms as transforms

from . import consts

# Embedding size (input of the classification layer) for the models that are given it;
# 'vgg' covers every VGG depth. Other models get None.
CIFAR_HIDDEN_SIZES = {'lenet5': 400, 'lenetzhu': 768, 'vgg': 512}
MNIST_HIDDEN_SIZES = {'lenet5': 256, 'lenetzhu': 588}
IMAGENET_HIDDEN_SIZES = {'vgg': 7 * 7 * 512}

IMAGENET_CLASSES = 1000
IMAGENET_RESIZE = 256
IMAGENET_CROP = 224
CROP_PADDING = 4


class SmallDataset(NamedTuple):
    """A torchvision dataset of small images."""
    torchvision_class: type
    image_size: int
    num_classes: int
    channels: int
    mean: tuple
    std: tuple
    hidden_sizes: dict
    grayscale: bool = False  # convert the images to one channel first


SMALL_DATASETS = {
    'CIFAR10': SmallDataset(torchvision.datasets.CIFAR10, 32, 10, 3, consts.cifar10_mean, consts.cifar10_std,
                            CIFAR_HIDDEN_SIZES),
    'CIFAR100': SmallDataset(torchvision.datasets.CIFAR100, 32, 100, 3, consts.cifar100_mean, consts.cifar100_std,
                             CIFAR_HIDDEN_SIZES),
    'MNIST': SmallDataset(torchvision.datasets.MNIST, 28, 10, 3, consts.mnist_mean, consts.mnist_std,
                          MNIST_HIDDEN_SIZES),
    'MNIST_GRAY': SmallDataset(torchvision.datasets.MNIST, 28, 10, 1, consts.mnist_mean, consts.mnist_std,
                               MNIST_HIDDEN_SIZES, grayscale=True),
}


def get_dataset(dataset, aug=True, data_path='data',
                normalize=True, split='all', model='lenet5'):
    """Return (image shape, number of classes, channels, embedding size, data) for a dataset name.

    data is (train set, validation set), or only one of them for split 'train' or 'val'.
    """
    path = os.path.expanduser(data_path)
    if dataset in SMALL_DATASETS:
        info = SMALL_DATASETS[dataset]
        shape_img = (info.image_size, info.image_size)
        num_classes, channel = info.num_classes, info.channels
        hidden = _hidden_size(model, info.hidden_sizes)
        trainset, validset = _build_small_dataset(info, path, aug, normalize)
    elif dataset == 'ImageNet':
        shape_img = (IMAGENET_CROP, IMAGENET_CROP)
        num_classes, channel = IMAGENET_CLASSES, 3
        hidden = _hidden_size(model, IMAGENET_HIDDEN_SIZES)
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


def _hidden_size(model, hidden_sizes):
    if 'vgg' in model:
        return hidden_sizes.get('vgg')
    return hidden_sizes.get(model)


def _normalization(mean, std, normalize):
    return transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x)


def _build_small_dataset(info, data_path, augmentations=True, normalize=True):
    """Train and validation sets, normalised; random crops and flips for training if asked."""
    trainset = info.torchvision_class(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = info.torchvision_class(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    to_gray = [transforms.Grayscale(num_output_channels=1)] if info.grayscale else []
    transform = transforms.Compose(to_gray + [transforms.ToTensor(), _normalization(info.mean, info.std, normalize)])
    if augmentations:
        trainset.transform = transforms.Compose(to_gray + [
            transforms.RandomCrop(info.image_size, padding=CROP_PADDING),
            transforms.RandomHorizontalFlip(),
            transform])
    else:
        trainset.transform = transform
    validset.transform = transform
    return trainset, validset


def _build_imagenet(data_path, augmentations=True, normalize=True):
    """ImageNet train and validation sets, resized and centre-cropped (random crops for training if asked)."""
    trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())
    mean, std = consts.imagenet_mean, consts.imagenet_std

    transform = transforms.Compose([
        transforms.Resize(IMAGENET_RESIZE),
        transforms.CenterCrop(IMAGENET_CROP),
        transforms.ToTensor(),
        _normalization(mean, std, normalize)])
    if augmentations:
        trainset.transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGENET_CROP),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalization(mean, std, normalize)])
    else:
        trainset.transform = transform
    validset.transform = transform
    return trainset, validset
