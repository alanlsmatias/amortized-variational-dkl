import os
import gc
import time
import pickle
import torch
import gpytorch
import gpinfuser

import torch.nn as nn
import torch.optim as optim
import numpy as np
import neural_tangents as nt

from typing import List, Dict, Callable, Tuple
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader

from utils import (
    Dataset,
    _get_feature_extractor,
    _kmeans_inducing_points,
    _get_function_mean,
    _get_function_kernel,
    _get_likelihood,
    get_scheduler,
    get_dataset,
    set_seed,
    save_json
)

MODEL_NAME = 'gdkl'

def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(prog='model-eval-uci-datasets')
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset name')
    parser.add_argument('--epochs', dest='epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', default=256, type=int, help='minibatch size')
    parser.add_argument('--num-splits', dest='num_splits', default=5, type=int, help='number of dataset splits')
    parser.add_argument('--features-ratio', dest='features_ratio', default=0.5, type=float, help='features ratio (default=0.5)')
    parser.add_argument('--num-inducing', dest='num_inducing', default=300, type=int, help='number of inducing points')
    parser.add_argument('--layer-size', dest='layer_size', default=64, type=int, help='number of weights per layer')
    parser.add_argument('--num-layers', dest='num_layers', default=1, type=int, help='number of layers')
    parser.add_argument('--workers', dest='num_workers', default=0, type=int, help='num. workers')
    parser.add_argument('--beta', dest='beta', default=1.0, type=float, help='KL-divergence beta')
    parser.add_argument('--lr', dest='lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--scheduler', dest='scheduler', action='store_const', const=True, default=False, help='on/off flag for using lr scheduler')
    parser.add_argument('--cuda', dest='device', action='store_const', const='cuda', default='cpu', help='on/off flag for using cuda')
    parser.add_argument('--seed', dest='seed', type=int, default=32, help='random generators seed')

    return parser

class GDKLDataset(torch.utils.data.Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        self.indices = torch.arange(len(x))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i: int):
        return self.x[i], self.y[i], self.indices[i]

def InfiniteNetworkKernelFunction(
    hidden_dim: List[int],
    activation: Tuple=nt.stax.Gelu(),
    W_std: float=1.0,
    b_std: float=1.0
) -> Callable:
    layers = []
    for dim in hidden_dim[:-1]:
        layers.append(nt.stax.Dense(dim, W_std=W_std, b_std=b_std))
        layers.append(activation)
    layers.append(nt.stax.Dense(hidden_dim[-1], W_std=W_std, b_std=b_std))
    return nt.stax.serial(*layers)[-1]

class InfiniteNetworkKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        hidden_dim: List[int],
        activation: Tuple=nt.stax.Rbf(),
        W_std: float=1.5,
        b_std: float=0.5
    ):
        gpytorch.kernels.Kernel.__init__(self, has_lengthscale=False)
        self.kernel_fn = InfiniteNetworkKernelFunction(hidden_dim, activation, W_std, b_std)
        self.base_covariance_matrix = None
    
    def precompute_base_covariance_matrix(
        self,
        x: Tensor,
        batch_size: int=2048,
        dtype: type=torch.float,
        device: str='cpu'
    ):
        num_data = x.size(0)
        self.base_covariance_matrix = torch.zeros(num_data, num_data, dtype=dtype, device=device)
        for i in range(0, num_data, batch_size):
            i_max = min(i + batch_size, num_data)
            ith_batch = x[i:i_max].detach().cpu().numpy()
            for j in range(i, num_data, batch_size):
                j_max = min(j + batch_size, num_data)
                jth_batch = x[j:j_max].detach().cpu().numpy()
                temp_covariance_matrix = np.asarray(self.kernel_fn(ith_batch, jth_batch, 'nngp'))
                temp_covariance_matrix = torch.tensor(temp_covariance_matrix, dtype=dtype, device=device)
                self.base_covariance_matrix[i:i_max,j:j_max] = temp_covariance_matrix
                if i != j:
                    self.base_covariance_matrix[j:j_max, i:i_max] = temp_covariance_matrix.t()

    def forward(self, x1: Tensor, x2: Tensor, **kwargs) -> Tensor:
        if self.base_covariance_matrix is None:
            x1_array = x1.cpu().numpy()
            x2_array = x2.cpu().numpy()
            covar = np.asarray(self.kernel_fn(x1_array, x2_array, 'nngp'))
            covar = torch.tensor(covar, dtype=x1.dtype, device=x1.device)
            return covar
        return self.base_covariance_matrix[x1.ravel()][:, x2.ravel()]
    
class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.Likelihood
    ):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        function_mean = self.mean_module(x)
        function_covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(function_mean, function_covar)
    
class GuidedVariationalELBO(gpytorch.mlls.VariationalELBO):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        gplayer: gpytorch.Module,
        dkl: gpytorch.Module,
        nngp: gpytorch.models.ExactGP,
        num_data: int,
        num_tasks: int,
        beta: float=1.0,
        train_size: float=0.5,
        **kwargs
    ):
        super().__init__(likelihood, gplayer, num_data, beta, combine_terms=False)
        self.dkl = dkl
        self.nngp = nngp.eval()
        self.train_size = train_size
        self.is_regression_task = isinstance(self.nngp.likelihood, gpytorch.likelihoods.GaussianLikelihood)
        self.num_tasks = num_tasks

    def prepare_targets(self, targets: Tensor, alpha_epsilon: float=0.01, dtype: torch.dtype=torch.float) -> Tensor:
        alpha = alpha_epsilon * torch.ones(targets.shape[-1], self.num_tasks, device=targets.device, dtype=dtype)
        alpha[torch.arange(len(targets)), targets] = alpha[torch.arange(len(targets)), targets] + 1
        sigma2_i = torch.log(alpha.reciprocal() + 1.0)
        transformed_targets = torch.transpose(alpha.log() - 0.5 * sigma2_i, 0, 1)
        return transformed_targets
    
    def split_data(self, inputs: Tensor, targets: Tensor, indices: Tensor) -> Dict[str, Tensor]:
        train_size = int(inputs.size(0) * self.train_size)
        split_indices = torch.randperm(inputs.size(0))
        split_0 = split_indices[:train_size]
        split_1 = split_indices[train_size:]
        return {
            'x0': inputs[split_0],
            'y0': targets[split_0],
            'i0': indices[split_0],
            'x1': inputs[split_1],
            'y1': targets[split_1],
            'i1': indices[split_1]
        }
    
    def compute_elbo(
        self,
        x_train_nngp: Tensor,
        y_train_nngp: Tensor,
        x_test_nngp: Tensor,
        x_test: Tensor,
        y_test: Tensor
    ) -> Tensor:

        if not self.is_regression_task:
            y_train_nngp = self.prepare_targets(y_train_nngp, self.nngp.likelihood.alpha_epsilon)

        self.nngp.set_train_data(x_train_nngp, y_train_nngp, strict=False)
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            nngp_dist_f = self.nngp(x_test_nngp)
        
        dkl_dist_f = self.dkl(x_test)
        log_likelihood = self._log_likelihood_term(dkl_dist_f, y_test).div(dkl_dist_f.event_shape[0])
        
        if self.is_regression_task:
            prior = torch.distributions.Normal(loc=nngp_dist_f.mean, scale=nngp_dist_f.stddev)
            posterior = torch.distributions.Normal(loc=dkl_dist_f.mean, scale=dkl_dist_f.stddev)
        else:
            prior = gpytorch.distributions.MultivariateNormal(nngp_dist_f.mean.t(), torch.diag_embed(nngp_dist_f.variance.t()))
            posterior = gpytorch.distributions.MultivariateNormal(dkl_dist_f.mean, torch.diag_embed(dkl_dist_f.variance))

        kl_divergence = torch.distributions.kl\
            .kl_divergence(posterior, prior)\
            .mean()\
            .div(self.num_data / self.beta)

        return log_likelihood - kl_divergence
    
    def forward(self, inputs: Tensor, targets: Tensor, indices: Tensor) -> Tensor:
        splits = self.split_data(inputs, targets, indices)
        elbo0 = self.compute_elbo(splits['i0'], splits['y0'], splits['i1'], splits['x1'], splits['y1'])
        elbo1 = self.compute_elbo(splits['i1'], splits['y1'], splits['i0'], splits['x0'], splits['y0'])
        return self.train_size * elbo0 + (1 - self.train_size) * elbo1

def _config_dataloader(train_loader: DataLoader, test_loader: DataLoader):
    def get_dataloader(dataloader: DataLoader, shuffle: bool):
        return DataLoader(
            dataset=GDKLDataset(*dataloader.dataset.tensors),
            batch_size=dataloader.batch_size,
            shuffle=shuffle,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )
    return get_dataloader(train_loader, True), get_dataloader(test_loader, False)

def _sample_from_dataset(dataset: GDKLDataset, task: str='regression', train_size: float=0.1):
    x, y = dataset.indices, dataset.y
    train_size = int(train_size * x.size(0))
    stratify = None if task == 'regression' else y
    x_sample, _, y_sample, _ = train_test_split(x, y, train_size=train_size, stratify=stratify)
    return x_sample, y_sample

def _nngp(
    dataset: GDKLDataset,
    depth: int=2,
    num_tasks: int=1,
    task='classification',
    device: str='cpu'
):
    indices, targets = _sample_from_dataset(dataset, task)
    indices, targets = indices.to(device), targets.to(device)
    
    assert task in ('classification', 'regression')

    if task == 'classification':
        targets = targets.long()
        batch_shape = torch.Size([num_tasks])
        likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            targets=targets,
            alpha_epsilon=0.01,
            learn_additional_noise=True
        )
        likelihood.second_noise_covar.initialize(noise=0.2)
    else:
        batch_shape = torch.Size([])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=targets.var() * 0.1)

    mean_fn = gpytorch.means.ConstantMean(batch_shape=batch_shape)
    kernel_fn = InfiniteNetworkKernel([1] * depth, nt.stax.Gelu())
    kernel_fn.precompute_base_covariance_matrix(dataset.x.cpu(), device=device)
    kernel_fn = gpytorch.kernels.ScaleKernel(kernel_fn, batch_shape=batch_shape)
    kernel_fn.initialize(outputscale=1.0)

    model = GaussianProcess(indices, targets, mean_fn, kernel_fn, likelihood).to(device)
    criterion = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    return model, criterion, optimizer

def _sparse_dkl(
    dataset: Dataset,
    train_dataset: GDKLDataset,
    nngp: GaussianProcess,
    num_inducing: int,
    beta: float=1.0,
    train_size: float=0.5,
    features_ratio: float=0.5,
    hidden_dim: List[int]=[64],
    activation: nn.Module=nn.SiLU(),
    lr: float=0.005,
    weight_decay: float=0.0,
    device: str='cpu'
):
    num_features = int(dataset.num_features // features_ratio)
    hidden_dim.append(num_features)
    feature_extractor = _get_feature_extractor(dataset, hidden_dim, activation)

    with torch.no_grad():
        feature_extractor.eval()
        latent = feature_extractor(train_dataset.x)
    inducing_points, _, _ = _kmeans_inducing_points(latent, num_inducing)

    gplayer = gpinfuser.models.SVGP(
        mean_module=_get_function_mean(),
        covar_module=_get_function_kernel(num_features),
        inducing_points=inducing_points,
        num_tasks=dataset.num_tasks
    )
    likelihood = _get_likelihood(dataset)

    if dataset.task == 'regression':
        likelihood.initialize(noise=train_dataset.y.var() * 0.1)

    model = gpinfuser.models.DKL(feature_extractor, gplayer, likelihood).to(device)
    criterion = GuidedVariationalELBO(
        likelihood=likelihood,
        gplayer=model.gplayer,
        dkl=model,
        nngp=nngp,
        num_data=len(train_dataset),
        num_tasks=dataset.num_tasks,
        beta=beta,
        train_size=train_size
    )
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'weight_decay': weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': lr * 0.1},
        {'params': model.gplayer.variational_parameters()},
        {'params': model.likelihood.parameters()}
    ], lr=lr)

    return model, criterion, optimizer

@torch.no_grad()
def predict_batch(model, dataloader):
    model.eval()
    y_true, y_pred, y_vars = [], [], []
    for x, y, _ in dataloader:
        x = x.to(args.device)
        y_true.append(y.cpu())
        
        if dataset.task == 'regression':
            y_dist = model.likelihood(model(x))
            y_pred.append(y_dist.mean.squeeze().cpu())
            y_vars.append(y_dist.variance.squeeze().cpu())
        else:
            if isinstance(model, GaussianProcess):
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    f_dist = model(x)
                f_samples = f_dist.rsample(torch.Size([500])).cpu()
                exp_f_samples = f_samples.exp()
                probabilities = (exp_f_samples / exp_f_samples.sum(-2, keepdim=True)).mean(0)
                y_pred.append(probabilities.t())
            else:
                with gpytorch.settings.num_likelihood_samples(500):
                    y_dist = model.likelihood(model(x))
                y_pred.append(y_dist.probs.mean(0).squeeze().cpu())
    
    y_true = torch.cat(y_true).double()
    y_pred = torch.cat(y_pred).double()
    if dataset.task == 'regression':
        y_vars = torch.cat(y_vars).double()

    return y_true, y_pred, y_vars

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    set_seed(args.seed)

    default_dir = os.path.join('reports/', args.dataset)
    os.makedirs(default_dir, exist_ok=True)

    dataset_file = os.path.join(default_dir, 'dataset.pickle')
    if os.path.isfile(dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = get_dataset(args)
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)

    dataset.set_full_train_data()

    if dataset.task == 'regression':
        metrics = ['nll', 'rmse']
    else:
        metrics = ['nll', 'acc', 'ece', 'brier']

    default_dir = os.path.join(default_dir, MODEL_NAME)
    os.makedirs(default_dir, exist_ok=True)

    reports, targets = {}, {}
    early_stopping_window = args.epochs // 10
    validation_scoring = gpinfuser.metrics.reg_negative_log_likelihood \
        if dataset.task == 'regression' \
        else gpinfuser.metrics.clf_negative_log_likelihood

    num_epochs_nngp = 300
    num_epochs_sdkl = args.epochs

    for i, (train_loader, val_loader, test_loader) in enumerate(dataset):
        if dataset.task == 'regression':
            train_loader, test_loader, targets_prior = dataset.reconfig_regression_dataloaders(train_loader, test_loader)
        train_loader, test_loader = _config_dataloader(train_loader, test_loader)

        state_dict_path = os.path.join(default_dir, f'model_splits_{i}.pt')
        csv_writer = gpinfuser.utils.CSVWriter(os.path.join(default_dir, f'train_split_{i}.csv'))
        csv_writer.initialize(['epoch', 'elapsed_time', 'mll', 'val_score', 'step'] + metrics)

        # Train the NNGP model
        nngp, nngp_criterion, nngp_optimizer = _nngp(
            dataset=train_loader.dataset,
            num_tasks=dataset.num_classes,
            depth=args.num_layers + 1,
            task=dataset.task,
            device=args.device
        )

        for epoch in range(1, num_epochs_nngp + 1):
            nngp.train()
            start_time = time.time()
            
            function_dist = nngp(nngp.train_inputs[0])
            if dataset.task == 'classification':
                nngp_mll = nngp_criterion(function_dist, nngp.likelihood.transformed_targets).sum()
            else:
                nngp_mll = nngp_criterion(function_dist, nngp.train_targets)
            
            nngp_optimizer.zero_grad()
            (-nngp_mll).backward()
            nngp_optimizer.step()
            elapsed_time = time.time() - start_time

            csv_log = {'epoch': epoch, 'elapsed_time': elapsed_time, 'mll': nngp_mll.item()}
            csv_writer.step(csv_log)

            print(f'Epoch {epoch}: mll={csv_log["mll"]:.4f}')
        
        # Delete unused variables
        del function_dist, nngp_mll, nngp_criterion, nngp_optimizer

        # Train the DKL model
        dkl, dkl_criterion, dkl_optimizer = _sparse_dkl(
            dataset=dataset,
            train_dataset=train_loader.dataset,
            nngp=nngp,
            num_inducing=args.num_inducing,
            beta=args.beta,
            features_ratio=args.features_ratio,
            hidden_dim=[args.layer_size] * args.num_layers,
            activation=torch.nn.SiLU(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device
        )

        if args.scheduler:
            scheduler = get_scheduler(dkl_optimizer, args)

        for epoch in range(1, args.epochs + 1):

            dkl.train()
            pbar = gpinfuser.utils.Progbar(len(train_loader), prefix=f'Epoch {epoch}')
            total_elbo, num_data, elapsed_time = 0, 0, time.time()

            for x_batch, y_batch, i_batch in train_loader:
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)
                i_batch = i_batch.to(args.device).long()

                if dataset.task == 'classification':
                    y_batch = y_batch.long()

                elbo = dkl_criterion(x_batch, y_batch, i_batch)
                dkl_optimizer.zero_grad()
                (-elbo).backward()
                dkl_optimizer.step()

                total_elbo += (elbo.item() * len(x_batch))
                num_data += len(x_batch)

                postfix = {'elbo': round(total_elbo/num_data, 4)}
                pbar.set_postfix(postfix)
                pbar.step()

            elapsed_time = time.time() - elapsed_time

            if args.scheduler:
                scheduler.step()

            y_true, y_pred, y_vars = predict_batch(dkl, test_loader)
            csv_log = {'epoch': epoch, 'mll': total_elbo / num_data, 'elapsed_time': elapsed_time}
            postfix = {'mll': round(csv_log['mll'], 4)}

            if dataset.task == 'regression':
                y_true = y_true * targets_prior.stddev + targets_prior.mean
                y_pred = y_pred * targets_prior.stddev + targets_prior.mean
                y_vars = y_vars * targets_prior.variance

                csv_log['nll'] = gpinfuser.metrics.reg_negative_log_likelihood(y_true, y_pred, y_vars)
                csv_log['rmse'] = gpinfuser.metrics.root_mean_squared_error(y_true, y_pred)
                postfix['nll'] = round(csv_log['nll'], 4)
                postfix['rmse'] = round(csv_log['rmse'], 4)
            else:
                y_hat = (y_pred > 0.5).double() if y_pred.ndim == 1 else y_pred.argmax(1)
                csv_log['nll'] = gpinfuser.metrics.clf_negative_log_likelihood(y_true, y_pred)
                csv_log['acc'] = gpinfuser.metrics.accuracy_score(y_true, y_hat)
                csv_log['ece'] = gpinfuser.metrics.expected_calibration_error(y_true, y_pred)
                csv_log['brier'] = gpinfuser.metrics.brier_score(y_true, y_pred)
                postfix['nll'] = round(csv_log['nll'], 4)
                postfix['acc'] = round(csv_log['acc'], 4)
                postfix['ece'] = round(csv_log['ece'], 4)
                postfix['brier'] = round(csv_log['brier'], 4)

            pbar.set_postfix(postfix)
            pbar.step(0)
            pbar.close()
            csv_writer.step(csv_log)

            torch.save({
                'dkl': dkl.state_dict(),
                'nngp': nngp.state_dict(),
                'settings': args.__dict__
            }, state_dict_path)
        
        # Delete unused variables
        del nngp, dkl_criterion, dkl_optimizer, elbo, y_true, y_pred, y_vars

        dkl.load_state_dict(torch.load(state_dict_path)['dkl'])
        y_true, y_pred, y_vars = predict_batch(dkl, test_loader)
        if dataset.task == 'regression':
            targets[str(i)] = {
                'y_true': y_true.numpy().tolist(),
                'y_mean': y_pred.numpy().tolist(),
                'y_variance': y_vars.numpy().tolist(),
            }
            csv_log['nll'] = gpinfuser.metrics.reg_negative_log_likelihood(y_true, y_pred, y_vars)
            csv_log['rmse'] = gpinfuser.metrics.root_mean_squared_error(y_true, y_pred)
            reports[str(i)] = csv_log
        else:
            targets[str(i)] = {
                'y_true': y_true.numpy().tolist(),
                'y_prob': y_pred.numpy().tolist()
            }
            reports[str(i)] = csv_log
        
        targetpath = os.path.join(default_dir, 'targets.json')
        reportpath = os.path.join(default_dir, 'reports.json')
        save_json(targets, targetpath)
        save_json(reports, reportpath)

        # Delete unused variables
        del dkl, y_true, y_pred, y_vars
        gc.collect()
        torch.cuda.empty_cache()