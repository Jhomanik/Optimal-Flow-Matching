import matplotlib
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
from .tools import ewma, freeze, freeze_context

def plot_benchmark_emb(benchmark, emb_X, emb_Y, model, size=1024):
    with freeze_context(model):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=100, sharex=True, sharey=True)
        
        Y = benchmark.output_sampler.sample(size); Y.requires_grad_(True)
        X = benchmark.input_sampler.sample(size); X.requires_grad_(True)

        X_push = emb_Y.transform(model.push_nograd(X).cpu().numpy())
        Y = emb_Y.transform(Y.cpu().detach().numpy())
        X = emb_X.transform(X.cpu().detach().numpy())

        axes[0].scatter(X[:, 0], X[:, 1], edgecolors='black')
        axes[1].scatter(Y[:, 0], Y[:, 1], edgecolors='black')
        axes[2].scatter(X_push[:, 0], X_push[:, 1], c='peru', edgecolors='black')

        axes[0].set_title(r'Ground Truth Input $\mathbb{P}$', fontsize=12)
        axes[1].set_title(r'Ground Truth Output $\mathbb{Q}$', fontsize=12)
        axes[2].set_title(r'Forward Map $\nabla\psi_{\theta}\circ\mathbb{P}$', fontsize=12)
        
        fig.tight_layout()
        torch.cuda.empty_cache(); gc.collect()
        return fig, axes

def plot_W2(benchmark, W2):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
    ax.set_title('Wasserstein-2', fontsize=12)
    ax.plot(W2, c='blue', label='Estimated Cost')
    if hasattr(benchmark, 'linear_cost'):
        ax.axhline(benchmark.linear_cost, c='orange', label='Bures-Wasserstein Cost')
    if hasattr(benchmark, 'cost'):
        ax.axhline(benchmark.cost, c='green', label='True Cost')    
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig, ax

def plot_benchmark_metrics(benchmark, metrics, baselines=None):
    fig, axes = plt.subplots(1, 2, figsize=(12,4), dpi=100)
    cmap = {'identity' : 'red', 'linear' : 'orange', 'constant' : 'magenta'}

    def cos2cosdist(data, metric):
        if metric == 'cos':
            return 1. - np.asarray(data)
        return data
    
    if baselines is None:
        baselines = {}
    
    for i, metric in enumerate(["L2_UVP", "cos"]):

        axes.flatten()[i].set_title(metric, fontsize=12)
        axes.flatten()[i].semilogy(cos2cosdist(metrics[metric], metric), label='Fitted Transport')
        for baseline in cmap.keys():
            if not baseline in baselines.keys():
                continue
            axes.flatten()[i].axhline(cos2cosdist(baselines[baseline][metric], metric), label=f'{baseline} baseline', c=cmap[baseline])
            
        axes.flatten()[i].legend(loc='upper left')
    
    fig.tight_layout()
    return fig, axes
