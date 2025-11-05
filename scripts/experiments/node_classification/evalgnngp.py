import os
import math
import json
import torch
import gpytorch
import numpy as np

from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO

from gpinfuser.utils import Progbar, CSVWriter
from gpinfuser.metrics import (
    accuracy_score,
    clf_negative_log_likelihood,
    expected_calibration_error,
    brier_score
)

from models import get_svdkl, get_idsgp, get_avdkl, get_gnn
from utils import (
    get_args_parser,
    get_citation_dataset,
    get_optimizer,
    get_logger,
    set_seed,
    timeit
)

def build_models_settings(args, dataset, data):
    return {
        'in_features': data.num_features,
        'num_classes': dataset.num_classes,
        'args': args
    }

def get_model(model, kwargs):
    if model == 'svdkl':
        return get_svdkl(**kwargs)
    if model == 'idsgp':
        return get_idsgp(**kwargs)
    if model == 'avdkl':
        return get_avdkl(**kwargs)
    if model == 'gnn':
        return get_gnn(**kwargs)
    raise ValueError('model `%s` is not valid' % model)

def train_gnngp(run_id, args, data, logger, kwargs):
    if args.pre_train is not None:
        if args.pre_train == 'from_scratch':
            current_model = args.model
            args.model = 'gnn'
            _, _, gnn = train_gnn(run_id, args, data, logger, kwargs)
            args.model = current_model
        else:
            assert os.path.isdir(args.pre_train)
            gnn, _ = get_model('gnn', kwargs)
            state_dict_path = os.path.join(args.pre_train, f'{run_id}.pt')
            gnn.load_state_dict(torch.load(state_dict_path)['model'])
        kwargs['pre_trained_gnn'] = gnn.deep_gcn

    lr = args.lr
    epochs = args.epochs
    model, parameters = get_model(args.model, kwargs)
    optimizer, scheduler = get_optimizer(parameters, lr, epochs, args.scheduler)
    mll = VariationalELBO(model.likelihood, model.gplayer, data.train_mask.sum().item())

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'[{run_id}] num_parameters={num_parameters}')
    logger.info(f'[{run_id}] scheduler={args.scheduler}')
    logger.info(f'[{run_id}] device={args.device}\n')
    logger.info(f'[{run_id}] Training: tracking accuracy score ...')

    def step():
        model.train()
        optimizer.zero_grad()
        function_dist = model(data.x, edge_index=data.edge_index, mask=data.train_mask)
        with gpytorch.settings.num_likelihood_samples(args.training_samples):
            elbo = mll(function_dist, data.y[data.train_mask])
        (-elbo).backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return elbo.item()

    def evaluate():
        mask = torch.ones(data.size(0)).bool()
        y_dist = model.predict(data.x, edge_index=data.edge_index, mask=mask, num_samples=args.inference_samples)
        y_prob = y_dist.probs.mean(0)
        y_pred = y_prob.argmax(1)
        y_val = data.y[data.val_mask]
        y_test = data.y[data.test_mask]

        return {
            'val_acc': accuracy_score(y_val, y_pred[data.val_mask]),
            'val_nll': clf_negative_log_likelihood(y_val, y_prob[data.val_mask]),
            'test_acc': accuracy_score(y_test, y_pred[data.test_mask]),
            'test_nll': clf_negative_log_likelihood(y_test, y_prob[data.test_mask]),
            'test_ece': expected_calibration_error(y_test, y_prob[data.test_mask]),
            'test_brier': brier_score(y_test, y_prob[data.test_mask])
        }

    csv_writer = CSVWriter(os.path.join(rpath, f'{run_id}.csv'))
    csv_writer.initialize([
        'epoch',
        'elapsed_time',
        'eval_time',
        'elbo',
        'val_acc',
        'val_nll',
        'test_acc',
        'test_nll',
        'test_ece',
        'test_brier'
    ])

    best_val_score = math.inf
    best_test_score = -math.inf
    validation_steps = args.val_steps
    val_steps_counter = 0

    pbar = Progbar(args.epochs, prefix=f'[{run_id}]')
    for epoch in range(1, args.epochs + 1):
        elbo, step_time = timeit(step)
        eval_results, eval_time = timeit(evaluate)
        val_score = eval_results['val_nll']

        if val_score < best_val_score:
            val_steps_counter = 0
            best_val_score = val_score
            best_test_score = eval_results['test_acc']

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(rpath, f'{run_id}.pt'))

            logging_msg = f'[{run_id}] Epoch {epoch}: val_score={best_val_score:.4f}, '
            logging_msg += f'test_acc={best_test_score:.4f}'
            logger.info(logging_msg)
        else:
            val_steps_counter += 1

        pbar.set_postfix({
            'epoch': epoch,
            'elbo': round(elbo, 4),
            'val_score': round(best_val_score, 4),
            'test_acc': round(eval_results['test_acc'], 4),
            'best_test_acc': round(best_test_score, 4)
        })
        pbar.step()

        csv_writer.step({
            'epoch': epoch,
            'elapsed_time': step_time,
            'eval_time': eval_time,
            'elbo': elbo
        } | eval_results)

        if val_steps_counter == validation_steps:
            break

    pbar.close()
    csv_writer.close()

    logger.info(f'[{run_id}] Training finished.')
    logger.info(f'[{run_id}] val_nll={best_val_score:.4f}, test_acc={best_test_score:.4f}')

    return best_val_score, best_test_score, model

def train_dkl(run_id, args, data, logger, kwargs):
    if args.pre_train is not None:
        if args.pre_train == 'from_scratch':
            current_model = args.model
            args.model = 'gnn'
            _, _, gnn = train_gnn(run_id, args, data, logger, kwargs)
            args.model = current_model
        else:
            assert os.path.isdir(args.pre_train)
            gnn, _ = get_model('gnn', kwargs)
            state_dict_path = os.path.join(args.pre_train, f'{run_id}.pt')
            gnn.load_state_dict(torch.load(state_dict_path)['model'])
        kwargs['pre_trained_gnn'] = gnn.deep_gcn

    model, parameters = get_model(args.model, kwargs)
    optimizer, scheduler = get_optimizer(
        parameters=parameters,
        lr=args.lr,
        epochs=args.epochs,
        scheduler=args.scheduler
    )
    criterion = ExactMarginalLogLikelihood(model.likelihood, model)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'[{run_id}] num_parameters={num_parameters}')
    logger.info(f'[{run_id}] scheduler={args.scheduler}')
    logger.info(f'[{run_id}] device={args.device}\n')
    logger.info(f'[{run_id}] Training: tracking accuracy score ...')

    def step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        f = model(model.train_inputs[0], edge_index=data.edge_index, features=data.x)
        mll = criterion(f, model.train_targets).sum()
        (-mll).backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return mll.item()
    
    @torch.inference_mode()
    def evaluate():
        model.eval()
        x = torch.ones(data.size(0), device=args.device).bool().nonzero()

        y_prob = model.predict(x, edge_index=data.edge_index, features=data.x)
        y_pred = y_prob.argmax(1)
        y_val = data.y[data.val_mask]
        y_test = data.y[data.test_mask]

        return {
            'val_acc': accuracy_score(y_val, y_pred[data.val_mask]),
            'val_nll': clf_negative_log_likelihood(y_val, y_prob[data.val_mask]),
            'test_acc': accuracy_score(y_test, y_pred[data.test_mask]),
            'test_nll': clf_negative_log_likelihood(y_test, y_prob[data.test_mask]),
            'test_ece': expected_calibration_error(y_test, y_prob[data.test_mask]),
            'test_brier': brier_score(y_test, y_prob[data.test_mask])
        }
    
    csv_writer = CSVWriter(os.path.join(rpath, f'{run_id}.csv'))
    csv_writer.initialize([
        'epoch',
        'elapsed_time',
        'eval_time',
        'mll',
        'val_acc',
        'val_nll',
        'test_acc',
        'test_nll',
        'test_ece',
        'test_brier'
    ])

    best_val_score = math.inf
    best_test_score = -math.inf
    validation_steps = args.val_steps
    val_steps_counter = 0

    pbar = Progbar(args.epochs, prefix=f'[{run_id}]')
    for epoch in range(1, args.epochs + 1):
        mll, step_time = timeit(step)
        eval_results, eval_time = timeit(evaluate)
        val_score = eval_results['val_nll']

        if val_score < best_val_score:
            val_steps_counter = 0
            best_val_score = val_score
            best_test_score = eval_results['test_acc']

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(rpath, f'{run_id}.pt'))

            logging_msg = f'[{run_id}] Epoch {epoch}: val_nll={best_val_score:.4f}, '
            logging_msg += f'test_acc={best_test_score:.4f}'
            logger.info(logging_msg)
        else:
            val_steps_counter += 1

        pbar.set_postfix({
            'epoch': epoch,
            'mll': round(mll, 4),
            'val_score': round(best_val_score, 4),
            'test_acc': round(eval_results['test_acc'], 4),
            'best_test_acc': round(best_test_score, 4)
        })
        pbar.step()

        csv_writer.step({
            'epoch': epoch,
            'elapsed_time': step_time,
            'eval_time': eval_time,
            'mll': mll
        } | eval_results)

        if val_steps_counter == validation_steps:
            break

    pbar.close()
    csv_writer.close()

    logger.info(f'[{run_id}] Training finished.')
    logger.info(f'[{run_id}] val_nll={best_val_score:.4f}, test_acc={best_test_score:.4f}')

    return best_val_score, best_test_score, model

def train_gnn(run_id, args, data, logger, kwargs):
    model, parameters = get_model(args.model, kwargs)
    optimizer, scheduler = get_optimizer(parameters, args.lr, args.epochs, args.scheduler)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'[{run_id}] num_parameters={num_parameters}')
    logger.info(f'[{run_id}] scheduler={args.scheduler}')
    logger.info(f'[{run_id}] device={args.device}\n')
    logger.info(f'[{run_id}] Training: tracking accuracy score ...')

    def step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, edge_index=data.edge_index)
        loss = model.loss(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return loss.item()
    
    def evaluate():
        y_prob = model.predict(data.x, data.edge_index)
        y_pred = y_prob.argmax(1)
        y_val = data.y[data.val_mask]
        y_test = data.y[data.test_mask]

        return {
            'val_acc': accuracy_score(y_val, y_pred[data.val_mask]),
            'val_nll': clf_negative_log_likelihood(y_val, y_prob[data.val_mask]),
            'test_acc': accuracy_score(y_test, y_pred[data.test_mask]),
            'test_nll': clf_negative_log_likelihood(y_test, y_prob[data.test_mask]),
            'test_ece': expected_calibration_error(y_test, y_prob[data.test_mask]),
            'test_brier': brier_score(y_test, y_prob[data.test_mask])
        }
    
    csv_writer = CSVWriter(os.path.join(rpath, f'{run_id}.csv'))
    csv_writer.initialize([
        'epoch',
        'elapsed_time',
        'eval_time',
        'loss',
        'val_acc',
        'val_nll',
        'test_acc',
        'test_nll',
        'test_ece',
        'test_brier'
    ])

    best_val_score = math.inf
    best_test_score = -math.inf
    validation_steps = args.val_steps
    val_steps_counter = 0

    pbar = Progbar(args.epochs, prefix=f'[{run_id}]')
    for epoch in range(1, args.epochs + 1):
        loss, step_time = timeit(step)
        eval_results, eval_time = timeit(evaluate)
        val_score = eval_results['val_nll']

        if val_score < best_val_score:
            val_steps_counter = 0
            best_val_score = val_score
            best_test_score = eval_results['test_acc']

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(rpath, f'{run_id}.pt'))

            logging_msg = f'[{run_id}] Epoch {epoch}: val_nll={best_val_score:.4f}, '
            logging_msg += f'test_acc={best_test_score:.4f}'
            logger.info(logging_msg)
        else:
            val_steps_counter += 1

        pbar.set_postfix({
            'epoch': epoch,
            'loss': round(loss, 4),
            'val_score': round(best_val_score, 4),
            'test_acc': round(eval_results['test_acc'], 4),
            'best_test_acc': round(best_test_score, 4)
        })
        pbar.step()

        csv_writer.step({
            'epoch': epoch,
            'elapsed_time': step_time,
            'eval_time': eval_time,
            'loss': loss
        } | eval_results)

        if val_steps_counter == validation_steps:
            break

    pbar.close()
    csv_writer.close()

    logger.info(f'[{run_id}] Training finished.')
    logger.info(f'[{run_id}] val_nll={best_val_score:.4f}, test_acc={best_test_score:.4f}')

    return best_val_score, best_test_score, model

if __name__ == '__main__':
    args = get_args_parser().parse_args()

    assert args.model in ('svdkl', 'idsgp', 'avdkl', 'gnn')
    assert args.dataset in ('cora', 'citeseer', 'pubmed')

    if args.pre_train is not None:
        rpath = f'{__file__}/../reports/{args.model}_pre_trained/{args.dataset}'
    else:
        rpath = f'{__file__}/../reports/{args.model}/{args.dataset}'

    os.makedirs(rpath, exist_ok=True)
    logger = get_logger('run', os.path.join(rpath, 'run.log'))
    logger.info(f'Running `{args.model}` on `{args.dataset}` dataset ...')

    if args.model == 'gnn':
        train = train_gnn
    elif args.model in ('svdkl', 'idsgp', 'avdkl'):
        train = train_gnngp
    else:
        raise ValueError('model `%s` is not a valid model' % (args.model,))

    with open(os.path.join(rpath, 'settings.json'), 'w') as f:
        json.dump(args.__dict__, f)

    results = {'val_nll': [], 'test_acc': []}
    for run_id in range(1, args.num_runs + 1):
        torch.cuda.empty_cache()
        set_seed(run_id)

        dataset, data = get_citation_dataset(args)
        kwargs = build_models_settings(args, dataset, data)

        val_nll, test_acc, _ = train(run_id, args, data, logger, kwargs)
        results['val_nll'].append(val_nll)
        results['test_acc'].append(test_acc)

        running_mean_acc = np.mean(results['test_acc']) * 100
        running_std_acc = np.std(results['test_acc']) * 100

        print(f'[{run_id}] acc={running_mean_acc:.2f} +/- {running_std_acc:.2f}\n')
        logger.info(f'[{run_id}] acc={running_mean_acc:.2f} +/- {running_std_acc:.2f}\n')

    for handler in logger.handlers:
        handler.close()

    with open(os.path.join(rpath, 'results.json'), 'w') as f:
        json.dump(results, f)
