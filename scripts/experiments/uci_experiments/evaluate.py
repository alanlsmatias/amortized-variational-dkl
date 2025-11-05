import os
import gc
import time
import pickle
import math
import torch

from gpytorch.mlls import VariationalELBO
from gpinfuser.utils import Progbar, CSVWriter
from gpinfuser.mlls import LatentVariableVariationalELBO
from gpinfuser.metrics import (
    reg_negative_log_likelihood,
    root_mean_squared_error,
    clf_negative_log_likelihood,
    accuracy_score,
    expected_calibration_error,
    brier_score
)

from utils import (
    get_args_parser,
    set_seed,
    get_dataset,
    get_model,
    get_optimizer,
    get_scheduler,
    save_json
)

def train_one_epoch(epoch: int):
    model.train()
    pbar = Progbar(len(train_loader), prefix=f'Epoch {epoch}')
    train_loss, num_data, elapsed_time = 0, 0, time.time()
    for x, y in train_loader:
        x, y = x.to(args.device), y.to(args.device)

        out = model(x)
        if args.model == 'dlvkl':
            loss = -mll(*out, y)
        else:
            loss = -mll(out, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp_num_data = len(y)
        train_loss += (loss.item() * temp_num_data)
        num_data += temp_num_data

        postfix = {'loss': round(train_loss/num_data, 4)}
        pbar.set_postfix(postfix)
        pbar.step()
    elapsed_time = time.time() - elapsed_time
    return train_loss/num_data, elapsed_time, pbar

def predict_batch(dataloader):
    y_true, y_pred, y_vars = [], [], []
    for x, y in dataloader:
        x = x.to(args.device)
        y_true.append(y.cpu())
        f_dist = model.predict(x)
        if dataset.task == 'regression':
            y_pred.append(f_dist.mean.squeeze().cpu())
            y_vars.append(f_dist.variance.squeeze().cpu())
        else:
            if model.gplayer.num_tasks == 1:
                y_pred.append(f_dist.probs.squeeze().cpu())
            else:
                y_pred.append(f_dist.probs.mean(0).squeeze().cpu())
    
    y_true = torch.cat(y_true).double()
    y_pred = torch.cat(y_pred).double()
    if dataset.task == 'regression':
        y_vars = torch.cat(y_vars).double()

    return y_true, y_pred, y_vars

if __name__ == '__main__':
    # parse args and set rng seed
    args = get_args_parser().parse_args()
    set_seed(args.seed)

    # config/load the dataset
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

    default_dir = os.path.join(default_dir, args.model)
    os.makedirs(default_dir, exist_ok=True)

    reports, targets = {}, {}
    early_stopping_window = args.epochs // 10
    validation_scoring = reg_negative_log_likelihood \
        if dataset.task == 'regression' \
        else clf_negative_log_likelihood
    
    for i, (train_loader, val_loader, test_loader) in enumerate(dataset):
        model, parameters = get_model(
            model_name=args.model,
            dataset=dataset,
            train_loader=train_loader,
            num_inducing=args.num_inducing,
            features_ratio=args.features_ratio,
            hidden_dim=[args.layer_size] * args.num_layers,
            nonlinearity=torch.nn.SiLU(),
            saturation=args.saturation,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device
        )

        if dataset.task == 'regression':
            train_loader, test_loader, targets_prior = dataset.reconfig_regression_dataloaders(train_loader, test_loader)
            model.likelihood.initialize(noise=train_loader.dataset.tensors[1].var() * 0.1)

        optimizer = get_optimizer(parameters, args)
        if args.scheduler:
            scheduler = get_scheduler(optimizer, args)

        if args.model == 'dlvkl':
            mll = LatentVariableVariationalELBO(
                likelihood=model.likelihood,
                model=model.gplayer,
                num_data=len(train_loader.dataset),
                beta=args.dlvkl_beta
            )
        else:
            mll = VariationalELBO(model.likelihood, model.gplayer, len(train_loader.dataset))

        csv_writer = CSVWriter(os.path.join(default_dir, f'train_split_{i}.csv'))
        csv_writer.initialize(['epoch', 'elapsed_time', 'elbo', 'val_score'] + metrics)

        best_val_score = math.inf
        validation_steps = 0
        state_dict_path = os.path.join(default_dir, f'model_splits_{i}.pt')

        for epoch in range(1, args.epochs + 1):
            loss, elapsed_time, pbar = train_one_epoch(epoch)

            if args.scheduler:
                scheduler.step()

            # Compute test scores
            y_true, y_pred, y_vars = predict_batch(test_loader)
            csv_log = {'epoch': epoch, 'elapsed_time': elapsed_time, 'elbo': -loss}
            postfix = {'loss': round(loss, 4)}

            if dataset.task == 'regression':
                y_true = y_true * targets_prior.stddev + targets_prior.mean
                y_pred = y_pred * targets_prior.stddev + targets_prior.mean
                y_vars = y_vars * targets_prior.variance

                csv_log['nll'] = reg_negative_log_likelihood(y_true, y_pred, y_vars)
                csv_log['rmse'] = root_mean_squared_error(y_true, y_pred)
                postfix['nll'] = round(csv_log['nll'], 4)
                postfix['rmse'] = round(csv_log['rmse'], 4)
            else:
                y_hat = (y_pred > 0.5).double() if y_pred.ndim == 1 else y_pred.argmax(1)
                csv_log['nll'] = clf_negative_log_likelihood(y_true, y_pred)
                csv_log['acc'] = accuracy_score(y_true, y_hat)
                csv_log['ece'] = expected_calibration_error(y_true, y_pred)
                csv_log['brier'] = brier_score(y_true, y_pred)
                postfix['nll'] = round(csv_log['nll'], 4)
                postfix['acc'] = round(csv_log['acc'], 4)
                postfix['ece'] = round(csv_log['ece'], 4)
                postfix['brier'] = round(csv_log['brier'], 4)

            # Compute validation score for ealy-stopping
            if val_loader is not None:
                y_val, y_val_pred, y_val_var = predict_batch(val_loader)
                val_score = validation_scoring(y_val, y_val_pred, y_val_var)
            else:
                val_score = -math.inf
            csv_log['val_score'] = val_score
            postfix['val_score'] = round(val_score, 4)

            pbar.set_postfix(postfix)
            pbar.step(0)
            pbar.close()
            csv_writer.step(csv_log)

            if val_score < best_val_score or val_loader is None:
                best_val_score = val_score
                validation_steps = 0
                torch.save({
                    'model': model.state_dict(),
                    'settings': args.__dict__
                }, state_dict_path)
            else:
                validation_steps += 1

            if validation_steps == early_stopping_window:
                break

        csv_writer.close()

        # Compute final test scores
        model.load_state_dict(torch.load(state_dict_path)['model'])
        y_true, y_pred, y_vars = predict_batch(test_loader)
        if dataset.task == 'regression':
            targets[str(i)] = {
                'y_true': y_true.numpy().tolist(),
                'y_mean': y_pred.numpy().tolist(),
                'y_variance': y_vars.numpy().tolist(),
            }
            csv_log['nll'] = reg_negative_log_likelihood(y_true, y_pred, y_vars)
            csv_log['rmse'] = root_mean_squared_error(y_true, y_pred)
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

        del model

        gc.collect()
        torch.cuda.empty_cache()