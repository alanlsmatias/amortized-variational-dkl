import os
import time
import pickle
import argparse
import torch
import gpytorch
import gpinfuser

from typing import Tuple
from torch import Tensor
from utils import get_dataset, set_seed, save_json

MODEL_NAME = "gp"

def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GP experiments on UCI datasets")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--num_splits", type=int, default=5, help="Number of splits")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--alpha_epsilon", type=float, default=1e-2, help="Dirichlet alpha epsilon")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser

class GPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        mean: gpytorch.means.Mean,
        kernel: gpytorch.kernels.Kernel
    ):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @torch.no_grad()
    def predict(self, x: Tensor, num_samples: int=512):
        self.eval()
        with gpytorch.settings.fast_pred_var():
            function_dist = self(x)
            if isinstance(self.likelihood, gpytorch.likelihoods.DirichletClassificationLikelihood):
                f_samples = function_dist.rsample(torch.Size([num_samples])).exp()
                return (f_samples / f_samples.sum(-2, keepdim=True)).mean(0).t()
            else:
                return self.likelihood(function_dist)

def _gp(
    train_x: Tensor,
    train_y: Tensor,
    num_tasks: int=1,
    alpha_epsilon: float=1e-2,
    lr: float=0.01,
    task="regression",
    device: str="cpu"
) -> Tuple[GPModel, gpytorch.mlls.ExactMarginalLogLikelihood, torch.optim.Optimizer]:
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    assert task in ["regression", "classification"]

    if task == "regression":
        batch_shape = torch.Size([])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=train_y.var() * 0.1)
    else:
        train_y = train_y.long()
        batch_shape = torch.Size([num_tasks])
        likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            targets=train_y,
            alpha_epsilon=alpha_epsilon,
            learn_additional_noise=True
        )
        likelihood.noise_covar.initialize(noise=0.2)

    mean = gpytorch.means.ZeroMean(batch_shape=batch_shape)
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            ard_num_dims=train_x.size(1),
            batch_shape=batch_shape
        ),
        batch_shape=batch_shape
    )
    kernel.outputscale = 1.0
    kernel.base_kernel.lengthscale = 1.0

    model = GPModel(train_x, train_y, likelihood, mean, kernel)
    criterion = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer

def train_gp(
    split: int,
    model: GPModel,
    criterion: gpytorch.mlls.ExactMarginalLogLikelihood,
    optimizer: torch.optim.Optimizer,
    train_x: Tensor,
    train_y: Tensor,
    epochs: int=1000,
    device: str="cpu"
) -> None:
    
    model.train()
    
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    csv_writer = gpinfuser.utils.CSVWriter(os.path.join(default_dir, f"train_split_{split}.csv"))
    csv_writer.initialize(["epoch", "elapsed_time", "mll"])
    pbar = gpinfuser.utils.Progbar(epochs, prefix="Training")

    for epoch in range(epochs):
        start_time = time.time()
        optimizer.zero_grad()
        output = model(train_x)
        mll = criterion(output, train_y).sum()
        (-mll).backward()
        optimizer.step()
        elapsed_time = time.time() - start_time

        csv_log = {"epoch": epoch, "elapsed_time": elapsed_time, "mll": mll.item()}
        csv_writer.step(csv_log)

        pbar.set_postfix(csv_log)
        pbar.step()

    csv_writer.close()
    pbar.close()

@torch.no_grad()
def evaluate_gp(
    model: GPModel,
    test_x: Tensor,
    test_y: Tensor,
    targets_prior: torch.distributions.Normal=None,
    task: str="regression",
    device: str="cpu"
):
    
    assert task in ["regression", "classification"]
    
    model.eval()

    test_x = test_x.to(device)
    test_y = test_y.to(device)

    if task == "regression":
        targets_dist = model.predict(test_x)
        y_true = test_y * targets_prior.stddev + targets_prior.mean
        y_pred = targets_dist.mean * targets_prior.stddev + targets_prior.mean
        y_vars = targets_dist.variance * targets_prior.variance

        targets = {
            "y_true": y_true.cpu().numpy().tolist(),
            "y_mean": y_pred.cpu().numpy().tolist(),
            "y_variances": y_vars.cpu().numpy().tolist()
        }
        results = {
            'nll': gpinfuser.metrics.reg_negative_log_likelihood(y_true, y_pred, y_vars),
            'rmse': gpinfuser.metrics.root_mean_squared_error(y_true, y_pred)
        }
    else:
        y_prob = model.predict(test_x)
        y_pred = (y_prob > 0.5).double() if y_prob.ndim == 1 else y_prob.argmax(1)
        targets = {
            "y_true": test_y.cpu().numpy().tolist(),
            "y_prob": y_prob.cpu().numpy().tolist(),
            "y_pred": y_pred.cpu().numpy().tolist()
        }
        results = {
            'nll': gpinfuser.metrics.clf_negative_log_likelihood(test_y, y_prob),
            'acc': gpinfuser.metrics.accuracy_score(test_y, y_pred),
            'ece': gpinfuser.metrics.expected_calibration_error(test_y, y_prob),
            'brier': gpinfuser.metrics.brier_score(test_y, y_prob)
        }
    
    return targets, results

def get_device(cuda: bool) -> str:
    return "cuda" if cuda and torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    
    default_dir = os.path.join("reports", args.dataset)
    os.makedirs(default_dir, exist_ok=True)

    dataset_file = os.path.join(default_dir, "dataset.pickle")
    if os.path.isfile(dataset_file):
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = get_dataset(args)
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)

    dataset.set_full_train_data()

    if dataset.task == "regression":
        metrics = ["nll", "rmse"]
    else:
        metrics = ["nll", "acc", "ece", "brier"]
    
    default_dir = os.path.join(default_dir, MODEL_NAME)
    os.makedirs(default_dir, exist_ok=True)

    reports, targets = {}, {}

    for i, (train_loader, val_loader, test_loader) in enumerate(dataset):
        if dataset.task == "regression":
            train_loader, test_loader, targets_prior = dataset\
                .reconfig_regression_dataloaders(train_loader, test_loader)
        train_x, train_y = train_loader.dataset.tensors
        test_x, test_y = test_loader.dataset.tensors

        model, criterion, optimizer = _gp(
            train_x,
            train_y,
            num_tasks=dataset.num_tasks,
            alpha_epsilon=args.alpha_epsilon,
            lr=args.lr,
            task=dataset.task,
            device=get_device(args.cuda)
        )

        train_gp(
            i,
            model,
            criterion,
            optimizer,
            train_x,
            train_y,
            args.epochs,
            device=get_device(args.cuda)
        )

        ith_targets, ith_reports = evaluate_gp(
            model,
            test_x,
            test_y,
            targets_prior=targets_prior,
            task=dataset.task,
            device=get_device(args.cuda)
        )

        targets[str(i)] = ith_targets
        reports[str(i)] = ith_reports

        torch.save({
            "model": model.state_dict(),
            "settings": args.__dict__
        }, os.path.join(default_dir, f"model_split_{i}.pt"))

        save_json(targets, os.path.join(default_dir, "targets.json"))
        save_json(reports, os.path.join(default_dir, "reports.json"))