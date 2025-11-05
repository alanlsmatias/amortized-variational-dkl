import os
import logging
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models import resnet18
from torchvision.datasets import (
    CIFAR10, CIFAR100, FGVCAircraft, Flowers102, OxfordIIITPet
)

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'num_train_data': 50000
    },
    'cifar100': {
        'num_classes': 100,
        'num_train_data': 50000
    },
    'aircraft': {
        'num_classes': 100,
        'num_train_data': 6667
    },
    'flowers': {
        'num_classes': 102,
        'num_train_data': 2040
    },
    'oxfordpets': {
        'num_classes': 37,
        'num_train_data': 3680
    }
}

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=None)
        self.num_features = resnet.inplanes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def config_cifar(self):
        self.features[0] = Conv2dNormActivation(3, 64, 3, 1, 1)
        self.reset_parameters()
        return self

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.flatten(out)
        return out

def get_args_parser(name=''):
    import argparse

    parser = argparse.ArgumentParser(prog=name)
    
    # feature extractor, dataset and cuda settings
    parser.add_argument('--feature-extractor', dest='feature_extractor', type=str, help='neural network to be used as feature extractor')
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset name (fasion_mnist, cifar10, cifar100)')
    parser.add_argument('--train-size', dest='train_size', type=float, default=0.0, help='dataset train size rate')
    parser.add_argument('--cuda', dest='device', action='store_const', const='cuda', default='cpu', help='on/off flag for using cuda')
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='RNG seed (default=0)')
    
    # dataloaders settings
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=0, help='number of dataloader workers')
    parser.add_argument('--persistent-workers', dest='persistent_workers', action='store_const', const=True, default=False, help='on/off flag for persistent workers')
    parser.add_argument('--epochs', dest='epochs', default=300, type=int, help='number of total epochs')

    # optimizer settings
    parser.add_argument('--opt', dest='optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr', dest='lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-momentum', dest='lr_momentum', type=float, default=0.9, help='learning rate momentum')
    parser.add_argument('--weight-decay', dest='lr_weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--scheduler', dest='scheduler', type=str, default='cosineannealinglr', help='learning rate scheduler')
    parser.add_argument('--lr-step', dest='lr_step', type=int, default=30, help='learning rate scheduler step size')
    parser.add_argument('--lr-gamma', dest='lr_gamma', type=float, default=0.1, help='learning rate scheduler gamma')
    parser.add_argument('--warmup-epochs', dest='lr_warmup_epochs', type=int, default=5, help='learning rate warmup epochs')
    parser.add_argument('--warmup-decay', dest='lr_warmup_decay', type=float, default=0.01, help='learning rate warmup decay')
    parser.add_argument('--clip-grad-norm', dest='clip_grad_norm', type=float, default=0.0, help='max grad norm')

    # extra augmentation settings
    parser.add_argument('--hfprob', dest='hfprob', type=float, default=0.5, help='random horizontal flip probability (default=0.5)')
    parser.add_argument('--reprob', dest='reprob', type=float, default=0.0, help='random erasing probability (default=0.0)')
    parser.add_argument('--mixup-alpha', dest='mixup_alpha', type=float, default=0.0, help='MixUp alpha')
    parser.add_argument('--cutmix-alpha', dest='cutmix_alpha', type=float, default=0.0, help='CutMix alpha')
    parser.add_argument('--tawide', dest='tawide', action='store_const', const=True, default=False, help='on/off flag for using trivial augmentation wide')

    return parser

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_feature_extractor(feature_extractor):
    if feature_extractor == 'resnet':
        return ResNet()
    raise ValueError('feature extractor `%s` is not valid' % feature_extractor)

def get_dataloaders(args):
    if args.num_workers == 0:
        assert not args.persistent_workers
    else:
        assert args.num_workers <= 6
    assert args.dataset in (
        'cifar10',
        'cifar100',
        # 'aircraft',
        # 'flowers',
        'oxfordpets'
    )

    if args.dataset == 'cifar10':
        mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    elif args.dataset == 'cifar100':
        mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    elif args.dataset == 'aircraft':
        mean, std = [0.491, 0.520, 0.543], [0.224, 0.216, 0.245]
    elif args.dataset == 'flowers':
        mean, std = [0.513, 0.416, 0.339], [0.294, 0.249, 0.289]
    elif args.dataset == 'oxfordpets':
        mean, std = [0.493, 0.445, 0.396], [0.259, 0.253, 0.260]

    # augmentations
    train_transform, test_transform = [v2.PILToTensor()], [v2.PILToTensor()]
    interpolation = v2.InterpolationMode.BILINEAR
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform.append(v2.RandomCrop(32, padding=4))
    else:
        train_transform.append(v2.RandomCrop(224))
        test_transform.append(v2.CenterCrop(224))
    if args.hfprob:
        train_transform.append(v2.RandomHorizontalFlip(p=args.hfprob))
    if args.tawide:
        train_transform.append(v2.TrivialAugmentWide(interpolation=interpolation))
    train_transform.extend([
        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean, std)
    ])
    test_transform.extend([
        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean, std)
    ])
    if args.reprob:
        train_transform.append(v2.RandomErasing(p=args.reprob))
    train_transform = v2.Compose(train_transform)
    test_transform = v2.Compose(test_transform)

    # load dataset
    datapath = f'{__file__}/../../../../datasets/image_classification/'
    if args.dataset == 'cifar10':
        datapath = os.path.join(datapath, 'cifar10')
        train_set = CIFAR10(root=datapath, train=True, transform=train_transform)
        test_set = CIFAR10(root=datapath, train=False, transform=test_transform)
    if args.dataset == 'cifar100':
        datapath = os.path.join(datapath, 'cifar100')
        train_set = CIFAR100(root=datapath, train=True, transform=train_transform)
        test_set = CIFAR100(root=datapath, train=False, transform=test_transform)
    if args.dataset == 'oxfordpets':
        datapath = os.path.join(datapath, 'oxford_iiit_pet')
        train_set = OxfordIIITPet(root=datapath, split='trainval', transform=train_transform)
        test_set = OxfordIIITPet(root=datapath, split='test', transform=test_transform)
    if args.dataset == 'aircraft':
        datapath = os.path.join(datapath, 'fgvc_aircraft')
        train_set = FGVCAircraft(root=datapath, split='trainval', transform=train_transform)
        test_set = FGVCAircraft(root=datapath, split='test', transform=test_transform)
    if args.dataset == 'flowers':
        datapath = os.path.join(datapath, 'flowers102')
        train_set = Flowers102(root=datapath, split='train', transform=train_transform)
        val_set = Flowers102(root=datapath, split='val')
        test_set = Flowers102(root=datapath, split='test', transform=test_transform)
        train_set._image_files.extend(val_set._image_files)
        train_set._labels.extend(val_set._labels)

    if args.train_size:
        assert args.dataset in ('cifar10', 'cifar100')
        x, _, y, _ = train_test_split(
            train_set.data,
            train_set.targets,
            train_size=args.train_size,
            stratify=train_set.targets
        )
        train_set.data = x
        train_set.targets = y

    # cutmix and mixup augmentation
    cutmixup = None
    if args.cutmix_alpha or args.mixup_alpha:
        num_classes = DATASETS[args.dataset]['num_classes']
        cutmixup = []
        if args.cutmix_alpha:
            cutmixup.append(v2.CutMix(alpha=args.cutmix_alpha, num_classes=num_classes))
        if args.mixup_alpha:
            cutmixup.append(v2.MixUp(alpha=args.mixup_alpha, num_classes=num_classes))
        cutmixup = v2.RandomChoice(cutmixup)

    # build dataloaders
    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=args.persistent_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, cutmixup

def get_optimizer(parameters, args):
    if args.optimizer == 'adam':
        return optim.Adam(parameters, lr=args.lr)
    if args.optimizer == 'sgd':
        return optim.SGD(
            params=parameters,
            lr=args.lr,
            weight_decay=0.0,
            momentum=args.lr_momentum
        )
    if args.optimizer == 'adamw':
        return optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    raise ValueError(f'optimizer `{args.optimizer}` is not valid')

def get_scheduler(optimizer, args):
    if args.scheduler == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_gamma
        )
    if args.scheduler == 'multisteplr':
        epochs = args.epochs - args.lr_warmup_epochs
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[int(0.5 * epochs), int(0.75 * epochs)],
            gamma=args.lr_gamma
        )
    elif args.scheduler == 'cosineannealinglr':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs - args.lr_warmup_epochs
        )
    else:
        raise ValueError(f'scheduler `{args.scheduler}` is not valid')
    
    if args.lr_warmup_epochs:
        warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=args.lr_warmup_decay,
            total_iters=args.lr_warmup_epochs
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_lr_scheduler, lr_scheduler],
            milestones=[args.lr_warmup_epochs]
        )

    return lr_scheduler

def get_logger(name: str, filepath: str):
    _format = '%(asctime)s, %(name)s: %(message)s'
    _dateformat = '%Y.%m.%d - %H:%M:%S'

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.isfile(filepath):
        os.remove(filepath)

    formatter = logging.Formatter(fmt=_format, datefmt=_dateformat)
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
