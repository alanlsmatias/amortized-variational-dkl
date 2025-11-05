import os
import time
import torch
import torch.nn as nn

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

def get_model(args):
    num_classes = DATASETS[args.dataset]['num_classes']

    feature_extractor = get_feature_extractor(args.feature_extractor)
    if args.dataset in ['cifar10', 'cifar100']:
        feature_extractor = feature_extractor.config_cifar()
    num_features = feature_extractor.num_features

    model = nn.Sequential(feature_extractor, nn.Linear(num_features, num_classes))
    model.to(args.device)
    parameters = [{'params': model.parameters(), 'weight_decay': args.lr_weight_decay}]

    return model, parameters

def train_one_epoch(epoch, clip_grad_norm=0.0):
    model.train()
    pbar = Progbar(len(train_dataloader), prefix=f'Epoch {epoch}')
    train_loss, num_data, elapsed_time = 0, 0, time.time()
    for x, y in train_dataloader:
        x, y = x.to(args.device), y.to(args.device)

        if cutmixup is not None:
            x, y = cutmixup(x, y)
        
        loss = criterion(model(x), y)

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
    scheduler.step()
    elapsed_time = time.time() - elapsed_time
    return train_loss/num_data, elapsed_time, pbar

@torch.inference_mode()
def evaluate():
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for x, y in test_dataloader:
        x, y = x.to(args.device), y.to(args.device)
        temp_y_prob = model(x).softmax(-1)
        y_true.append(y)
        y_prob.append(temp_y_prob)
        y_pred.append(temp_y_prob.argmax(-1))
    
    y_true = torch.cat(y_true)
    y_prob = torch.cat(y_prob)
    y_pred = torch.cat(y_pred)

    acc1 = accuracy_score(y_true, y_pred)
    acc5 = topk_accuracy_score(y_true, y_prob, k=3)
    nll = clf_negative_log_likelihood(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    brier = brier_score(y_true, y_prob)

    return acc1, acc5, nll, ece, brier

if __name__ == '__main__':
    args = get_args_parser('RunEvalImageClf').parse_args()
    set_seed(args.seed)

    mainpath = f'{__file__}/../reports/deepnn/{args.feature_extractor}/{args.dataset}/{args.seed}'
    os.makedirs(mainpath, exist_ok=True)
    logger = get_logger('run', os.path.join(mainpath, 'run.log'))
    logger.info(f'Running DeepNN ({args.feature_extractor.upper()}) on {args.dataset.upper()} dataset ...\n')

    for key, value in args.__dict__.items():
        logger.info(f'{key}={value}')

    model, parameters = get_model(args)
    train_dataloader, test_dataloader, cutmixup = get_dataloaders(args)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'num_parameters={num_parameters}')

    optimizer = get_optimizer(parameters, args)
    scheduler = get_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss()

    csv_writer = CSVWriter(os.path.join(mainpath, 'train.csv'))
    csv_writer.initialize([
        'epoch',
        'elapsed_time',
        'loss',
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
            'loss': round(loss, 3),
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
            'loss': loss,
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
