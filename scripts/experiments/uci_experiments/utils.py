import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood

from gpinfuser.nn import MLP, Variational
from gpinfuser.models import (
    GaussianProcess,
    SVDKL,
    DLVKL,
    IDSGP,
    AVDKL,
    SVGP,
    AmortizedSVGP,
    GridInterpolationSVGP,
    FeaturesVariationalDistribution
)

REG_DS = (
    'protein',
    'kegg_directed',
    'kegg_undirected',
    '3d_road'
)
CLF_DS = (
    'magic_gamma_telescope',
    'htru2',
    'letter',
    'bank_marketing',
    'crop',
    'skin_segmentation'
)

class Dataset:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        task: str,
        batch_size: int=128,
        num_workers: int=0,
        num_splits: int=5
    ):
    
        assert task in ('regression', 'classification')
        if task == 'regression':
            splits = KFold(n_splits=num_splits, shuffle=True).split(x)
            self.num_classes = None
            self.num_tasks = 1
        else:
            splits = StratifiedKFold(n_splits=num_splits, shuffle=True).split(x, y)
            splits = list(splits)
            y = LabelEncoder().fit_transform(y)
            self.num_classes = y.max() + 1
            self.num_tasks = self.num_classes

        nested_splits = []
        for train_idx, test_idx in splits:
            stratify = y[train_idx] if task == 'classification' else None
            inner_split = train_test_split(train_idx, test_size=0.2, stratify=stratify)
            inner_split.extend([test_idx])
            nested_splits.append(inner_split)

        self.x = x
        self.y = y
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = nested_splits
        self.num_features = x.shape[1]
        self.use_full_train_data = False

    def set_full_train_data(self):
        self.use_full_train_data = True
    
    def reconfig_regression_dataloaders(self, train_loader: DataLoader, test_loader: DataLoader):
        x_train, y_train = train_loader.dataset.tensors
        x_test, y_test = test_loader.dataset.tensors

        targets_prior = torch.distributions.Normal(y_train.mean(), y_train.std())
        y_train = (y_train - targets_prior.mean) / targets_prior.stddev
        y_test = (y_test - targets_prior.mean) / targets_prior.stddev

        def get_dataloader(x, y, shuffle):
            return DataLoader(
                dataset=TensorDataset(x, y),
                batch_size=train_loader.batch_size,
                shuffle=shuffle,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory
            )
        
        return (
            get_dataloader(x_train, y_train, True),
            get_dataloader(x_test, y_test, False),
            targets_prior
        )

    def __iter__(self):
        for train_idx, val_idx, test_idx in self.splits:
            x_train = self.x[train_idx]
            y_train = self.y[train_idx]
            x_val = self.x[val_idx]
            y_val = self.y[val_idx]
            x_test = self.x[test_idx]
            y_test = self.y[test_idx]

            if self.use_full_train_data:
                x_train = np.concatenate([x_train, x_val], axis=0)
                y_train = np.concatenate([y_train, y_val])

            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            train_set = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
            test_set = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float())
            
            train_loader = DataLoader(
                dataset=train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            test_loader = DataLoader(
                dataset=test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

            val_loader = None
            if not self.use_full_train_data:
                x_val = scaler.transform(x_val)
                val_set = TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).float())
                val_loader = DataLoader(
                    dataset=val_set,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True
                )

            yield train_loader, val_loader, test_loader

def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(prog='model-eval-uci-datasets')
    parser.add_argument('--model', dest='model', type=str, help='model name (svgp, svdkl, idsgp, avdkl)')
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset name')
    parser.add_argument('--epochs', dest='epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--warmup-epochs', dest='warmup_epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('--batch-size', dest='batch_size', default=256, type=int, help='minibatch size')
    parser.add_argument('--num-splits', dest='num_splits', default=5, type=int, help='number of dataset splits')
    parser.add_argument('--features-ratio', dest='features_ratio', default=0.5, type=float, help='features ratio (default=0.5)')
    parser.add_argument('--num-inducing', dest='num_inducing', default=300, type=int, help='number of inducing points')
    parser.add_argument('--layer-size', dest='layer_size', default=64, type=int, help='number of weights per layer')
    parser.add_argument('--num-layers', dest='num_layers', default=1, type=int, help='number of layers')
    parser.add_argument('--saturation', dest='saturation', default='sigmoid', type=str, help='amortization saturation function (default=sigmoid)')
    parser.add_argument('--workers', dest='num_workers', default=0, type=int, help='num. workers')
    parser.add_argument('--lr', dest='lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--scheduler', dest='scheduler', action='store_const', const=True, default=False, help='on/off flag for using lr scheduler')
    parser.add_argument('--cuda', dest='device', action='store_const', const='cuda', default='cpu', help='on/off flag for using cuda')
    parser.add_argument('--seed', dest='seed', type=int, default=32, help='random generators seed')
    parser.add_argument('--dlvkl-beta', dest='dlvkl_beta', type=float, default=0.1, help='DLVKL beta value')

    return parser

def read_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        content = json.load(f)
    return content

def save_json(content: Dict[str, Any], filepath: str):
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=4)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_dataset(args):
    current_dir = os.path.dirname(__file__)
    datapath = '%s/../../../datasets/uci/%s/%s/data.csv'
    task = 'regression' if args.dataset in REG_DS else 'classification'
    datapath = datapath % (current_dir, task, args.dataset)

    df_dataset = pd.read_csv(datapath)
    dataset = Dataset(
        x=df_dataset.drop(columns='target').values,
        y=df_dataset['target'].values,
        task=task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_splits=args.num_splits
    )

    return dataset

def get_optimizer(parameters, args):
    return optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

def _get_function_mean():
    return ZeroMean()

def _get_function_kernel(num_features: int):
    kernel = ScaleKernel(RBFKernel(ard_num_dims=num_features))
    kernel.outputscale = 1
    kernel.base_kernel.lengthscale = 1
    return kernel

def _get_likelihood(dataset: Dataset):
    if dataset.task == 'regression':
        return GaussianLikelihood()
    return SoftmaxLikelihood(num_classes=dataset.num_tasks, mixing_weights=None)
    
def _kmeans_inducing_points(x: torch.Tensor, num_inducing: int):
    x = x.numpy()
    kmeans = KMeans(n_clusters=num_inducing, n_init=10).fit(x)
    inducing_points = kmeans.cluster_centers_
    return torch.tensor(inducing_points).float()

def _get_feature_extractor(
    dataset: Dataset,
    hidden_dim: List[int]=[64],
    nonlinearity: nn.Module=nn.SiLU()
) -> MLP:
    feature_extractor = MLP(
        in_features=dataset.num_features,
        hidden_dim=hidden_dim,
        nonlinearity=nonlinearity,
        norm_layer=nn.BatchNorm1d
    )
    return feature_extractor

def get_svgp(
    dataset: Dataset,
    train_loader: DataLoader,
    num_inducing: int=512,
    device: str='cpu',
    *args, **kwargs
):
    inputs = train_loader.dataset.tensors[0]
    gplayer = SVGP(
        mean_module=_get_function_mean(),
        covar_module=_get_function_kernel(dataset.num_features),
        inducing_points=_kmeans_inducing_points(inputs, num_inducing),
        num_tasks=dataset.num_tasks
    )
    likelihood = _get_likelihood(dataset)
    model = GaussianProcess(gplayer, likelihood).to(device)
    return model, model.parameters()

def get_svdkl(
    dataset: Dataset,
    num_inducing: int,
    features_ratio: float=0.5,
    hidden_dim: List[int]=[64],
    nonlinearity: nn.Module=nn.SiLU(),
    lr: float=0.005,
    weight_decay: float=0.0,
    device: str='cpu',
    *args, **kwargs
):
    class SVDKLRegression(SVDKL):
        def forward(self, x):
            out = self.feature_extractor(x)
            return self.gplayer(out)
    
    num_features = int(dataset.num_features // features_ratio)
    hidden_dim.append(num_features)
    feature_extractor = _get_feature_extractor(dataset, hidden_dim, nonlinearity)
    if dataset.task == 'classification':
        gplayer = GridInterpolationSVGP(
            mean_module=_get_function_mean(),
            covar_module=_get_function_kernel(None),
            num_inducing=num_inducing,
            num_tasks=num_features,
            grid_bounds=[-10, 10]
        )
        likelihood = SoftmaxLikelihood(num_features, dataset.num_tasks)
        model = SVDKL(feature_extractor, gplayer, likelihood).to(device)
    else:
        gplayer = SVGP(
            mean_module=_get_function_mean(),
            covar_module=_get_function_kernel(None),
            inducing_points=torch.randn(num_inducing, num_features),
            num_tasks=dataset.num_tasks
        )
        likelihood = _get_likelihood(dataset)
        model = SVDKLRegression(feature_extractor, gplayer, likelihood).to(device)

    parameters = [
        {'params': model.feature_extractor.parameters(), 'weight_decay': weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': lr * 0.1},
        {'params': model.gplayer.variational_parameters()},
        {'params': model.likelihood.parameters()},
    ]
    return model, parameters

def get_dlvkl(
    dataset: Dataset,
    train_loader: DataLoader,
    num_inducing: int=512,
    features_ratio: float=0.5,
    hidden_dim: List[int]=[64],
    nonlinearity: nn.Module=nn.SiLU(),
    lr: float=0.005,
    weight_decay: float=0.0,
    device: str='cpu',
    *args, **kwargs
):
    num_features = int(dataset.num_features // features_ratio)
    hidden_dim.append(2 * num_features)
    mlp = _get_feature_extractor(dataset, hidden_dim, nonlinearity)
    feature_extractor = FeaturesVariationalDistribution(mlp, prior_noise=0.1)

    with torch.no_grad():
        feature_extractor.eval()
        latent, _, _ = feature_extractor(train_loader.dataset.tensors[0])
    inducing_points = _kmeans_inducing_points(latent, num_inducing)

    gplayer = SVGP(
        mean_module=_get_function_mean(),
        covar_module=_get_function_kernel(num_features),
        inducing_points=inducing_points,
        num_tasks=dataset.num_tasks
    )
    likelihood = _get_likelihood(dataset)
    model = DLVKL(feature_extractor, gplayer, likelihood).to(device)

    parameters = [
        {'params': model.feature_extractor.parameters(), 'weight_decay': weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': lr * 0.1},
        {'params': model.gplayer.variational_parameters()},
        {'params': model.likelihood.parameters()}
    ]

    return model, parameters

def get_idsgp(
    dataset: Dataset,
    num_inducing: int=3,
    hidden_dim: List[int]=[64],
    weight_decay: float=0.0,
    device: str='cpu',
    *args, **kwargs
):
    feature_extractor = _get_feature_extractor(dataset, hidden_dim, nn.Sigmoid())
    variational_estimator = Variational(
        in_features=hidden_dim[-1],
        num_tasks=dataset.num_tasks,
        num_features=dataset.num_features,
        num_inducing=num_inducing,
        saturation=nn.Sigmoid()
    )
    gplayer = AmortizedSVGP(
        mean_module=_get_function_mean(),
        covar_module=_get_function_kernel(None),
        num_inducing=num_inducing,
        num_tasks=dataset.num_tasks
    )
    likelihood = _get_likelihood(dataset)
    model = IDSGP(feature_extractor, variational_estimator, gplayer, likelihood).to(device)

    parameters = [
        {'params': model.feature_extractor.parameters(), 'weight_decay': weight_decay},
        {'params': model.variational_estimator.parameters(), 'weight_decay': weight_decay},
        {'params': model.gplayer.hyperparameters()},
        {'params': model.likelihood.parameters()}
    ]
    return model, parameters

def get_avdkl(
    dataset: Dataset,
    num_inducing: int,
    features_ratio: float=0.5,
    hidden_dim: List[int]=[64],
    nonlinearity: nn.Module=nn.SiLU(),
    saturation: nn.Module=nn.Sigmoid(),
    lr: float=0.005,
    weight_decay: float=0.0,
    device: str='cpu',
    *args, **kwargs
):
    assert saturation in ('sigmoid', 'tanh')
    
    num_features = int(dataset.num_features // features_ratio)
    hidden_dim.append(num_features)
    feature_extractor = _get_feature_extractor(dataset, hidden_dim, nonlinearity)
    variational_estimator = Variational(
        in_features=num_features,
        num_tasks=dataset.num_tasks,
        num_features=num_features,
        num_inducing=num_inducing,
        saturation=nn.Sigmoid() if saturation == 'sigmoid' else nn.Tanh()
    )
    gplayer = AmortizedSVGP(
        mean_module=_get_function_mean(),
        covar_module=_get_function_kernel(None),
        num_inducing=num_inducing,
        num_tasks=dataset.num_tasks
    )
    likelihood = _get_likelihood(dataset)
    model = AVDKL(feature_extractor, variational_estimator, gplayer, likelihood).to(device)

    parameters = [
        {'params': model.feature_extractor.parameters(), 'weight_decay': weight_decay},
        {'params': model.variational_estimator.parameters(), 'weight_decay': weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': lr * 0.1},
        {'params': model.likelihood.parameters()},
    ]

    return model, parameters

def get_model(model_name, **kwargs):
    if model_name == 'svgp':
        return get_svgp(**kwargs)
    if model_name == 'svdkl':
        return get_svdkl(**kwargs)
    if model_name == 'dlvkl':
        return get_dlvkl(**kwargs)
    if model_name == 'idsgp':
        return get_idsgp(**kwargs)
    if model_name == 'avdkl':
        return get_avdkl(**kwargs)
    raise ValueError('model `%s` is not valid (svgp, svdkl, idsgp, avdkl)' % model_name)
