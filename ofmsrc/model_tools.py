import torch
import torch.nn as nn
from tqdm import tqdm
from IPython.display import clear_output
import torch.nn.functional as F
from .icnn import GradNNGeneric
from .inverse import (
    ofm_inverse,
    ofm_inverse_grad_norms
)
from .inverse import (
    InvMethodParamsGeneric,
    TorchoptimInvParams,
    TorchlbfgsInvParams
)
from .ihv import ofm_ihv
from .ihv import (
    IHVMethodParamsGeneric,
    BruteforceIHVParams
)
from .ood import OOD
from typing import Union, Tuple, Dict, Optional
import time


def id_pretrain_model(
    model, sampler, lr=1e-3, n_max_iterations=2000, batch_size=1024, loss_stop=1e-5, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    for it in tqdm(range(n_max_iterations), disable = not verbose):
        X = sampler.sample((batch_size,))
        if len(X.shape) == 1:
            X = X.view(-1, 1)
        X.requires_grad_(True)
        loss = F.mse_loss(model.push(X), X)
        loss.backward()
        
        opt.step()
        opt.zero_grad() 
        model.convexify()
        
        if verbose:
            if it % 100 == 99:
                clear_output(wait=True)
                print('Loss:', loss.item())
            
            if loss.item() < loss_stop:
                clear_output(wait=True)
                print('Final loss:', loss.item())
                break
    return model


def ofm_forward(
        model: GradNNGeneric,
        X0: torch.Tensor,
        t: torch.Tensor
) -> torch.Tensor :
    if len(t.shape) == 1:
        t = t.unsqueeze(1)
    assert t.size(0) == X0.size(0)
    t = t.to(X0)
    Y = (1. - t) * X0 + t * model.push_nograd(X0)
    return Y.detach()

def ofm_loss(
        model: GradNNGeneric,
        X0: torch.Tensor,
        X1: torch.Tensor,
        t: torch.Tensor,
        inverse_params: InvMethodParamsGeneric,
        ihv_params: IHVMethodParamsGeneric,
        ood: OOD = OOD(),
        tol_inverse_border: Optional[float] = None,
        eps: float = 1e-5,
        reduction: str = 'mean',
        stats: Dict = dict()
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Computes OFM loss.
    Returns: 
      `loss_proxy` (permits gradient calculation)
      `true_loss` (value of the true ofm loss)
    See https://arxiv.org/abs/2403.13117 for the details
    '''
    elapsed_times = dict()
    t_loss = time.perf_counter()
    assert reduction in ['mean', 'sum']
    reduction_func = lambda x: torch.mean(x)
    if reduction == 'sum':
        reduction_func = lambda x: torch.sum(x)

    assert X0.shape == X1.shape
    t = t.squeeze()
    assert len(t.shape) == 1
    assert len(X0.shape) == 2
    assert X0.size(0) == t.size(0)
    t = t.to(X0)

    Xt = X0*(1-t[:,None]) + X1*t[:,None]
    
    inverse_stats = dict()
    t_inverse = time.perf_counter()
    Z0 = ofm_inverse(model, Xt, t, inverse_params, inverse_stats)
    elapsed_times['inverse'] = time.perf_counter() - t_inverse
    assert Z0.shape == X0.shape
    
    # applying ood:
    mask = ood(Z0)
    stats['ood_ratio'] = mask.sum().item()/len(mask)
    
    # applying inverse border
    if tol_inverse_border is not None:
        mask_norms = ofm_inverse_grad_norms(model, Z0, Xt, t) < tol_inverse_border
        stats['tol_inverse_ratio'] = mask_norms.sum().item()/len(mask_norms)
        mask = mask & mask_norms
    
    if mask.sum().item() < 2:
        # dummy loss
        loss = model.push(Z0).sum() * 0
        true_loss = torch.tensor(0.).to(X0)
        elapsed_times['ihv'] = 0
    else:
        X0, Z0, t = X0[mask], Z0[mask], t[mask]

        v = 2. * (X0 - Z0) / (t[:, None] + eps)

        t_ihv = time.perf_counter()
        lhs = ofm_ihv(model, Z0, v, t, ihv_params) # (bs, dim)
        elapsed_times['ihv'] = time.perf_counter() - t_ihv

        rhs = model.push(Z0)

        assert lhs.shape == rhs.shape
        loss = reduction_func((lhs * rhs).sum(dim = 1))
        # true loss
        true_loss = reduction_func((v * v).sum(1))
    elapsed_times['loss'] = time.perf_counter() - t_loss
    stats['times'] = elapsed_times
    stats['inverse'] = inverse_stats
    return loss, true_loss
