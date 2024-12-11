from collections import defaultdict
from tqdm import tqdm
from typing import List, Union

import torch
from torch import optim
import wandb
import os
import gc

from .scheduler import (
    TrainingSchedulerGeneric,
    TrainingSchedulerSM_Mixin,
    TrainingSchedulerWandB_Mixin,
    TrainingSchedulerModelsSaver_Mixin
)

def clean_resources(*tsrs):
    del tsrs
    gc.collect()
    torch.cuda.empty_cache()


def train_epoch(
    n_epoch, 
    model, 
    train_loader, 
    optimizer, 
    use_cuda,
    loss_key='total', 
    conditional=False,
    schedulers: List[TrainingSchedulerGeneric] = [TrainingSchedulerGeneric(),]
):
    model.train()

    for n_batch, x in enumerate(train_loader):
        if use_cuda:
            if not conditional:
                x = x.cuda()
            else:
                assert len(x) == 2
                x = [x[0].cuda(), x[1].cuda()]
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        # losses = {key: val.detach() for key, val in losses.items()}
        # clean_resources()
        for scheduler in schedulers:
            scheduler.on_batch_optim_step(epoch=n_epoch, batch=n_batch)
        optimizer.step()
        for scheduler in schedulers:
            scheduler.on_batch_train_end(epoch=n_epoch, batch=n_batch, losses=losses, data=x)
    
    for scheduler in schedulers:
        scheduler.on_epoch_train_end(epoch=n_epoch)


def eval_model(
    n_epoch, 
    model, 
    data_loader, 
    use_cuda,
    conditional=False, 
    schedulers: List[TrainingSchedulerGeneric] =[TrainingSchedulerGeneric(),]
):
    model.eval()
    stats = defaultdict(float)
    ds_length = 0
    with torch.no_grad():
        for x in data_loader:
            if use_cuda:
                if not conditional:
                    x = x.cuda()
                else:
                    assert len(x) == 2
                    x = [x[0].cuda(), x[1].cuda()]
            losses = model.loss(x)
            x_shape = x[0].shape[0] if conditional else x.shape[0]
            ds_length += x_shape
            for k, v in losses.items():
                stats[k] += v.item() * x_shape

        for k in stats.keys():
            stats[k] /= ds_length
        
        for scheduler in schedulers:
            scheduler.on_epoch_eval_end(epoch=n_epoch, losses=stats)


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr=None,
    optimizer=None,
    adam_betas=(0.9, 0.999),
    use_tqdm=False,
    use_cuda=False,
    loss_key='total_loss',
    conditional=False,
    scheduler: Union[TrainingSchedulerGeneric, List[TrainingSchedulerGeneric]] = TrainingSchedulerGeneric()
):
    schedulers = [scheduler,] if isinstance(scheduler, TrainingSchedulerGeneric) else scheduler
    if optimizer is None:
        assert lr is not None
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas)
        for sched in schedulers: 
            sched.optimizer = optimizer

    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.cuda()

    for epoch in forrange:
        model.train()
        train_epoch(
            epoch, model, train_loader, optimizer, 
            use_cuda, loss_key, conditional, schedulers)
        eval_model(
            epoch, model, test_loader, 
            use_cuda, conditional, schedulers)

