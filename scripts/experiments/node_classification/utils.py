import argparse
import logging
import os
import time
import random
import torch
import numpy as np

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from sklearn.feature_extraction.text import TfidfTransformer
from typing import List, Dict

DATASETS = {
    'cora': 'Cora',
    'citeseer': 'CiteSeer',
    'pubmed': 'PubMed'
}

class ToTfidf:
    def __call__(self, data):
        device = data.x.device
        x_tfidf = TfidfTransformer(smooth_idf=True).fit_transform(data.x.cpu()).todense()
        data.x = torch.tensor(x_tfidf).float().to(device)
        return data

def get_args_parser():
    parser = argparse.ArgumentParser(prog='GraphNodeClassification')

    # Model and dataset settings
    parser.add_argument('--model', dest='model', type=str, help='model name (svdkl, idsgp, avdkl, gnn)')
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset name (cora, citeseer, pubmed)')
    parser.add_argument(
        '--tfidf',
        dest='tfidf',
        action='store_const',
        const=True,
        default=False,
        help='on/off flag for using TFIDF encoding'
    )

    # Models settings
    parser.add_argument('--channels', dest='channels', type=int, default=64, help='number of channels (default=64)')
    parser.add_argument('--layers', dest='layers', type=int, default=64, help='number of convolutions (default=64)')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.6, help='dropout rate (default=0.6)')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1, help='initial connection factor (default=0.1)')
    parser.add_argument('--theta', dest='theta', type=float, default=0.0, help='identity mapping factor (default=0.0)')
    parser.add_argument('--inducing', dest='num_inducing', type=int, default=3, help='number of inducing points (default=3)')
    parser.add_argument('--matern', dest='matern', type=float, default=1.5, help='matern kernel nu (default=1.5)')
    parser.add_argument('--outputscale', dest='outputscale', type=float, default=1, help='initial outputscale (default=1)')
    parser.add_argument('--lengthscale', dest='lengthscale', type=float, default=1, help='initial lengthscale (default=1)')
    parser.add_argument('--grid-bounds', dest='grid_bounds', type=float, default=10, help='svdkl grid bounds (default=10)')

    # Evaluation and training settings
    parser.add_argument('--runs', dest='num_runs', type=int, default=30, help='number of independent weight initializations (default=30)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='number of epochs (default=1000)')
    parser.add_argument('--val-steps', dest='val_steps', type=int, default=100, help='number of validation steps (default=100)')
    parser.add_argument('--lr', dest='lr', type=float, default=0.005, help='learning rate (default=0.005)')
    parser.add_argument('--dense-weight-decay', dest='dense_weight_decay', type=float, default=0.0005, help='dense layer weight decay (default=0.0005)')
    parser.add_argument('--convs-weight-decay', dest='convs_weight_decay', type=float, default=0.01, help='convolution layers weight decay (default=0.01)')
    parser.add_argument(
        '--scheduler',
        dest='scheduler',
        action='store_const',
        const=True,
        default=False,
        help='on/off flag for using cosine annealing lr scheduler'
    )
    parser.add_argument(
        '--pre-train',
        dest='pre_train',
        type=str,
        default=None,
        help='path to the pre-trained GNN; use `from_scratch` for training from scratch (default=None)'
    )

    # Extra settings
    parser.add_argument('--training-samples', dest='training_samples', type=int, default=64, help='number of likelihood samples for training (default=64)')
    parser.add_argument('--inference-samples', dest='inference_samples', type=int, default=1024, help='number of likelihood samples for inference (default=1024)')
    parser.add_argument(
        '--cuda',
        dest='device',
        action='store_const',
        const='cuda',
        default='cpu',
        help='on/off flag for using cuda'
    )
    parser.add_argument('--seed', dest='seed', type=int, default=32, help='random generators seed (default=32)')

    return parser

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def timeit(fn):
    start_time = time.time()
    result = fn()
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_citation_dataset(args):
    assert args.dataset in ('cora', 'citeseer', 'pubmed')
    assert args.device in ('cpu', 'cuda')

    datapath = f'{__file__}/../../../../datasets'
    assert os.path.isdir(datapath)
    datapath = os.path.join(datapath, 'graph_node_clf',  DATASETS[args.dataset].lower())
    dataset = Planetoid(
        root=datapath,
        name=DATASETS[args.dataset],
        transform=ToTfidf() if args.tfidf else NormalizeFeatures()
    )
    data = dataset[0].to(args.device)
    return dataset, data

def get_optimizer(parameters: List[Dict], lr: float, epochs: int=None, scheduler: bool=False):
    optimizer = torch.optim.Adam(parameters, lr=lr)
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        return optimizer, lr_scheduler
    return optimizer, None

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