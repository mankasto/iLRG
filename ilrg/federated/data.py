"""Datasets, their normalization constants, and batch sampling.
Dataset loading modified from https://github.com/JonasGeiping/invertinggradients"""

import os
import random
from typing import NamedTuple

import torch
import torchvision
import torchvision.transforms as transforms

# Per-channel mean and std used to normalize each dataset
cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Classes of the custom_imbalanced and custom_balanced batch distributions (CIFAR-100 ids)
CUSTOM_DISTRIBUTION_CLASSES = (0, 18, 92)

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
    'CIFAR10': SmallDataset(torchvision.datasets.CIFAR10, 32, 10, 3, cifar10_mean, cifar10_std,
                            CIFAR_HIDDEN_SIZES),
    'CIFAR100': SmallDataset(torchvision.datasets.CIFAR100, 32, 100, 3, cifar100_mean, cifar100_std,
                             CIFAR_HIDDEN_SIZES),
    'MNIST': SmallDataset(torchvision.datasets.MNIST, 28, 10, 3, mnist_mean, mnist_std,
                          MNIST_HIDDEN_SIZES),
    'MNIST_GRAY': SmallDataset(torchvision.datasets.MNIST, 28, 10, 1, mnist_mean, mnist_std,
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
    mean, std = imagenet_mean, imagenet_std

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
