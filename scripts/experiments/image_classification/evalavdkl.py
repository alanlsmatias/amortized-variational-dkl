import os
import time
import math
import torch
import torch.nn as nn

from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO

from gpinfuser.nn import Variational
from gpinfuser.models import AVDKL, AmortizedSVGP
from gpinfuser.utils import Progbar, CSVWriter
from gpinfuser.metrics import (
    accuracy_score,
    topk_accuracy_score,
    clf_negative_log_likelihood,
    expected_calibration_error,
    brier_score
)

from utils import (
    DATASETS,
    get_feature_extractor,
    get_args_parser,
    get_dataloaders,
    get_optimizer,
    get_scheduler,
    set_seed,
    get_logger
)

def get_args_parser_2(name):
    parser = get_args_parser(name)
    parser.add_argument(
        '--kernel',
        dest='kernel',
        default='rbf',
        type=str,
        help='kernel function: rbf, matern12, matern32, matern52 (default=rbf)'
    )
    parser.add_argument(
        '--num-inducing',
        dest='num_inducing',
        default=15,
        type=int,
        help='number of inducing points'
    )
    parser.add_argument(
        '--pred-elbo',
        dest='pred_elbo',
        action='store_const',
        const=1,
        default=0,
        help='on/off flag for using the predictive log-likelihood'
    )
    parser.add_argument(
        '--saturation',
        dest='saturation',
        default='tanh',
        type=str,
        help='saturation function'
    )
    parser.add_argument(
        '--bottleneck',
        dest='bottleneck',
        default=8,
        type=int,
        help='bottleneck compression size'
    )
    return parser

def get_kernel(kernel, ard_num_dims=None):
    if kernel == 'rbf':
        kernel = RBFKernel(ard_num_dims=ard_num_dims)
    elif kernel == 'matern12':
        kernel = MaternKernel(nu=1/2, ard_num_dims=ard_num_dims)
    elif kernel == 'matern32':
        kernel = MaternKernel(nu=3/2, ard_num_dims=ard_num_dims)
    elif kernel == 'matern52':
        kernel = MaternKernel(nu=5/2, ard_num_dims=ard_num_dims)
    else:
        ValueError('kernel `%s` is not valie' % kernel)

    kernel.lengthscale = 15
    kernel = ScaleKernel(kernel)
    kernel.outputscale = 1

    return kernel

def get_model(args):
    num_classes = DATASETS[args.dataset]['num_classes']
    num_inducing = args.num_inducing

    feature_extractor = get_feature_extractor(args.feature_extractor)
    num_features = feature_extractor.num_features
    if args.dataset in ['cifar10', 'cifar100']:
        feature_extractor = feature_extractor.config_cifar()

    # variational parameters projection
    assert args.saturation in ('tanh', 'sigmoid')
    saturation_activation = nn.Tanh() if args.saturation == 'tanh' else nn.Sigmoid()
    bottleneck = num_features // args.bottleneck
    saturation = nn.Sequential(nn.Linear(num_features, bottleneck), saturation_activation)
    variational_estimator = Variational(
        in_features=bottleneck,
        num_tasks=num_classes,
        num_features=num_features,
        num_inducing=num_inducing,
        saturation=saturation
    )

    for m in variational_estimator.saturation.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    gplayer = AmortizedSVGP(
        mean_module=ZeroMean(),
        covar_module=get_kernel(args.kernel),
        num_inducing=num_inducing,
        num_tasks=num_classes
    )
    model = AVDKL(
        feature_extractor=feature_extractor,
        variational_estimator=variational_estimator,
        gplayer=gplayer,
        likelihood=SoftmaxLikelihood(num_classes=num_classes, mixing_weights=None)
    ).to(args.device)

    parameters = [
        {'params': model.feature_extractor.parameters(), 'weight_decay': args.lr_weight_decay},
        {'params': model.variational_estimator.parameters(), 'weight_decay': args.lr_weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': args.lr * 0.01},
        {'params': model.likelihood.parameters()}
    ]

    return model, parameters

def train_one_epoch(epoch, clip_grad_norm=0.0):
    model.train()
    pbar = Progbar(len(train_dataloader), prefix=f'Epoch {epoch}')
    train_loss, num_data, elapsed_time = 0, 0, time.time()
    for x, y in train_dataloader:
        x, y = x.to(args.device), y.to(args.device)

        if cutmixup is not None:
            x, y = cutmixup(x, y)

        loss = -mll(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        temp_num_data = len(x)
        train_loss += (loss.item() * temp_num_data)
        num_data += temp_num_data

        postfix = {'loss': round(train_loss/num_data, 4)}
        pbar.set_postfix(postfix)
        pbar.step()

    if scheduler is not None:
        scheduler.step()
    
    elapsed_time = time.time() - elapsed_time
    return train_loss/num_data, elapsed_time, pbar

@torch.inference_mode()
def evaluate():
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    for x, y in test_dataloader:
        y_hat = model.predict(x.to(args.device), num_samples=300).probs.mean(0).cpu()
        y_true.append(y)
        y_pred.append(y_hat.argmax(1))
        y_prob.append(y_hat)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_prob = torch.cat(y_prob)

    acc1 = accuracy_score(y_true, y_pred)
    acc5 = topk_accuracy_score(y_true, y_prob, k=3)
    nll = clf_negative_log_likelihood(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    brier = brier_score(y_true, y_prob)

    return acc1, acc5, nll, ece, brier

if __name__ == '__main__':
    args = get_args_parser_2('RunEvalImageClf').parse_args()
    set_seed(args.seed)

    mainpath = f'{__file__}/../reports/avdkl/{args.feature_extractor}/{args.dataset}/{args.seed}'
    os.makedirs(mainpath, exist_ok=True)
    logger = get_logger('run', os.path.join(mainpath, 'run.log'))
    logger.info(f'Running AVDKL ({args.feature_extractor.upper()}) on {args.dataset.upper()} dataset ...\n')

    for key, value in args.__dict__.items():
        logger.info(f'{key}={value}')

    model, parameters = get_model(args)
    train_dataloader, test_dataloader, cutmixup = get_dataloaders(args)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'num_parameters={num_parameters}')

    optimizer = get_optimizer(parameters, args)
    scheduler = get_scheduler(optimizer, args)
    mll = VariationalELBO(model.likelihood, model.gplayer, len(train_dataloader.dataset))

    csv_writer = CSVWriter(os.path.join(mainpath, 'train.csv'))
    csv_writer.initialize([
        'epoch',
        'elapsed_time',
        'elbo',
        'acc@1',
        'acc@3',
        'nll',
        'ece',
        'brier'
    ])

    for epoch in range(1, args.epochs + 1):
        loss, elapsed_time, pbar = train_one_epoch(epoch, args.clip_grad_norm)
        acc1, acc5, nll, ece, brier = evaluate()

        pbar.set_postfix({
            'elbo': round(-loss, 3),
            'acc@1': round(acc1, 3),
            'acc@3': round(acc5, 3),
            'nll': round(nll, 3),
            'ece': round(ece, 3),
            'brier': round(brier, 3)
        })
        pbar.step(0)
        pbar.close()

        csv_writer.step({
            'epoch': epoch,
            'elapsed_time': elapsed_time,
            'elbo': -loss,
            'acc@1': acc1,
            'acc@3': acc5,
            'nll': nll,
            'ece': ece,
            'brier': brier
        })

    csv_writer.close()

    for handler in logger.handlers:
        handler.close()

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args.__dict__
    }, os.path.join(mainpath, 'model.pt'))
