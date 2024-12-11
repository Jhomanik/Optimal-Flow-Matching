import torch
import torch.nn as nn
import pandas as pd
from typing import Optional, List
from copy import deepcopy

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def is_freezed(model: nn.Module) -> bool:
    req_grad = next(iter(model.parameters())).requires_grad
    train_mode = model.training
    if train_mode and req_grad:
        return False
    if (not train_mode) and (not req_grad):
        return True
    raise Exception('Inconsistent in freezing!')

class freeze_context:

    def __init__(self, model: nn.Module):
        self.model = model
    
    def __enter__(self):
        self.model_freezed_flag = is_freezed(self.model)
        freeze(self.model)
        return self.model
    
    def __exit__(self, *exc):
        if not self.model_freezed_flag:
            unfreeze(self.model)

class EMA:

    @staticmethod
    def update_average(model_tgt : nn.Module, model_src : nn.Module, beta: float):
        with torch.no_grad():
            params_src = model_src.parameters()
            params_tgt = model_tgt.parameters()
            for p_src, p_tgt in zip(params_src, params_tgt):
                assert p_src.shape == p_tgt.shape
                p_tgt.data.copy_(beta*p_tgt.data + (1. - beta)*p_src.data)

    def __init__(self, init_step : Optional[int] = None, betas: List[float] = [0.999, 0.99]):
        self.betas = betas
        self.init_step = init_step
        self._step = 0
    
    def __call__(self, model: nn.Module):
        if self.init_step is None:
            self.model_copy = model
        else:
            if self._step < self.init_step:
                pass
            if self._step == self.init_step:
                self.model_copies = [deepcopy(model) for _ in range(len(self.betas))]
            if self._step > self.init_step:
                for model_copy, beta in zip(self.model_copies, self.betas):
                    self.update_average(model_copy, model, beta)
        self._step += 1
    
    def get_model(self, beta: float) -> nn.Module:
        if self.init_step is None:
            return self.model_copy
        for i, _beta in enumerate(self.betas):
            if _beta == beta:
                return self.model_copies[i]
        raise Exception(f'beta = {beta} is not supported')

    
