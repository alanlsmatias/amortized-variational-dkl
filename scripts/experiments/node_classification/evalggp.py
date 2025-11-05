import os
import argparse
import warnings
import math
import json
import torch

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpinfuser.utils import Progbar, CSVWriter
from gpinfuser.metrics import (
    accuracy_score,
    clf_negative_log_likelihood
)

from models import get_ggp
from utils import (
    get_citation_dataset,
    get_optimizer,
    get_logger,
    set_seed,
    timeit
)

ALPHA_EPSILON = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
EPOCHS = 300
LR = 0.05

def get_args_parser():
    parser = argparse.ArgumentParser(prog='ggp-node-clf')
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset name (cora, citeseer, pubmed)')
    parser.add_argument('--norm', dest='norm', type=str, default='ggp', help='adj. matrix norm (ggp, gcn)')
    parser.add_argument('--K', dest='K', type=int, default=1, help='number of adj. matrix aggregations')
    parser.add_argument('--power', dest='power', type=float, default=3.0, help='polynomial kernel power')
    parser.add_argument(
        '--tfidf', dest='tfidf',
        action='store_const', const=True, default=False,
        help='on/off flag for using TFIDF encoding'
    )
    parser.add_argument(
        '--cuda', dest='device',
        action='store_const', const='cuda', default='cpu',
        help='on/off flag for using cuda'
    )
    parser.add_argument(
        '--seed', dest='seed', type=int, default=32,
        help='random generators seed'
    )
    return parser

def get_model(args, data, num_classes: int, alpha_epsilon: float):
    return get_ggp(
        x_train=data.train_mask.nonzero().to(args.device),
        y_train=data.y[data.train_mask].to(args.device),
        node_features=data.x,
        edge_index=data.edge_index,
        num_classes=num_classes,
        norm=args.norm,
        K=args.K,
        power=args.power,
        alpha_epsilon=alpha_epsilon,
        device=args.device
    )

def get_criterion(model, likelihood):
    return ExactMarginalLogLikelihood(likelihood, model)

def train(alpha_epsilon, args, data, num_classes, logger):
    model, parameters = get_model(args, data, num_classes, alpha_epsilon)
    optimizer, _ = get_optimizer(parameters, LR)
    criterion = get_criterion(model, model.likelihood)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'[{alpha_epsilon}] num_parameters={num_parameters}')
    logger.info(f'[{alpha_epsilon}] alpha_epsilon={alpha_epsilon}')
    logger.info(f'[{alpha_epsilon}] Training ...')

    def step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        f = model(model.train_inputs[0])
        mll = criterion(f, model.train_targets).sum()
        (-mll).backward()
        optimizer.step()
        return mll.item()
    
    @torch.inference_mode()
    def evaluate(x, y):
        model.eval()
        y_prob = model.predict(x).cpu()
        y_pred = y_prob.argmax(1)
        return {
            'acc': accuracy_score(y, y_pred),
            'nll': clf_negative_log_likelihood(y, y_prob)
        }
    
    y_train = data.y[data.train_mask]
    y_test = data.y[data.test_mask]
    x_test = data.test_mask.nonzero().to(args.device)

    csv_filename = str(alpha_epsilon).replace('.', '').replace('-', '')
    csv_writer = CSVWriter(os.path.join(rpath, f'{csv_filename}.csv'))
    csv_writer.initialize([
        'epoch',
        'elapsed_time',
        'mll',
        'train_acc',
        'train_nll',
        'test_acc',
        'test_nll'
    ])

    pbar = Progbar(EPOCHS, prefix=f'[{alpha_epsilon}]')
    for epoch in range(1, EPOCHS + 1):
        mll, step_time = timeit(step)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train_scores = evaluate(model.train_inputs[0], y_train)
            test_scores = evaluate(x_test, y_test)

        pbar.set_postfix({
            'epoch': epoch,
            'elapsed_time': round(step_time, 4),
            'mll': round(mll, 4),
            'train_nll': round(train_scores['nll'], 4),
            'test_acc': round(test_scores['acc'], 4)
        })
        pbar.step()

        csv_writer.step({
            'epoch': epoch,
            'elapsed_time': step_time,
            'mll': mll,
            'train_acc': train_scores['acc'],
            'train_nll': train_scores['nll'],
            'test_acc': test_scores['acc'],
            'test_nll': test_scores['nll']
        })

    pbar.close()
    csv_writer.close()

    train_nll, test_acc = train_scores['nll'], test_scores['acc']

    logger.info(f'[{alpha_epsilon}] Training finished.')
    logger.info(f'[{alpha_epsilon}] train_nll={train_nll:.4f}, test_acc={test_acc:.4f}')

    return train_nll, test_acc

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    set_seed(args.seed)

    rpath = '%s/../reports/gp/%s/%s' % (__file__, args.norm, args.dataset)
    os.makedirs(rpath, exist_ok=True)
    logger = get_logger('run', os.path.join(rpath, 'run.log'))
    logger.info('Running `gp|%s_norm` on `%s` dataset ...' % (args.norm, args.dataset))

    dataset, data = get_citation_dataset(args.dataset)

    best_train_nll, best_test_acc = math.inf, -math.inf
    results = {'train_nll': [], 'test_acc': []}
    for alpha_epsilon in ALPHA_EPSILON:
        train_nll, test_acc = train(
            alpha_epsilon, args, data, dataset.num_classes, logger
        )
        results['train_nll'].append(train_nll)
        results['test_acc'].append(test_acc)

        if train_nll < best_train_nll:
            best_train_nll = train_nll
            best_test_acc = test_acc

        print(f'[{alpha_epsilon}] acc={test_acc:.4f}\n')
        logger.info(f'[{alpha_epsilon}] acc={test_acc:.4f}\n')

    for handler in logger.handlers:
        handler.close()

    with open(os.path.join(rpath, 'results.json'), 'w') as f:
        json.dump(results, f)