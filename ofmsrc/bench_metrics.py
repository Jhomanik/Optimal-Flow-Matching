import numpy as np
import torch
import gc
from .tools import freeze_context

def score_fitted_map(benchmark, D, size=4096):
    '''Estimates L2-UVP and cosine metrics for transport map by the gradient of potential'''
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True); Y.requires_grad_(True)

    with freeze_context(D):
        X_push = D.push_nograd(X)
        
        with torch.no_grad():
            L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
            
            cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
            (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
            
        gc.collect(); torch.cuda.empty_cache() 
        return L2_UVP_fwd, cos_fwd

def score_baseline_map(benchmark, baseline='linear', size=4096):
    '''Estimates L2-UVP and cosine similarity metrics for the baseline transport map (identity, const or linear).'''
    assert baseline in ['identity', 'linear', 'constant']
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True)
    
    with torch.no_grad():
        if baseline == 'linear':  
            X_push = benchmark.linear_map_fwd(X)
        elif baseline == 'constant':
            X_push = torch.tensor(
                benchmark.output_sampler.mean.reshape(1, -1).repeat(size, 0),
                dtype=torch.float32
            ).to(X)
        elif baseline == 'identity':
            X_push = X

        if baseline == 'constant':
            L2_UVP_fwd = 100.
        else:
            L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()

        if baseline == 'identity':
            cos_fwd = 0.
        else:
            cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
            (np.sqrt(2 * benchmark.cost * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()

    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd, cos_fwd

def metrics_to_dict(L2_UVP_fwd, cos_fwd):
    return dict(L2_UVP=L2_UVP_fwd, cos=cos_fwd)
