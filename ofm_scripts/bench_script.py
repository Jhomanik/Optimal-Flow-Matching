import os, sys
sys.path.append("..")

# torch
import wandb
import torch
import random
import numpy as np
from datetime import datetime
import time
from numbers import Number

# icnn
from ofmsrc.icnn import DenseICNN

# discrete ot
from ofmsrc.discrete_ot import OTPlanSampler

from ofmsrc.distributions import StandardNormalSampler
from ofmsrc.bench_tools import train_identity_map

from collections import defaultdict
import numpy as np
import random
import string
import pickle
import argparse

from ofmsrc.model_tools import (
    ofm_forward,
    ofm_loss
)

from ofmsrc.tools import EMA

from ofmsrc.model_tools import (
    TorchlbfgsInvParams,
    TorchoptimInvParams,
    BruteforceIHVParams
)

from ofmsrc.bench_metrics import (
    score_fitted_map,
    metrics_to_dict,
    score_baseline_map
)

from ofmsrc.bench_plotters import (
    plot_W2,
    plot_benchmark_metrics
)

parser = argparse.ArgumentParser(
    description='ofm benchmark',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# genereal settings

# REQUIRED ARGUMENTS
parser.add_argument('--dim', type=int, help='problem dimensionality')
parser.add_argument('--gpu', type=int, help='device number')
parser.add_argument('--sp', type=str, help='sampler: "idp" or "mb" or "gt"')
parser.add_argument('--mb', type=int, help='minibatch OT batch size')
parser.add_argument('--wandb', action='store_const', const=True, default=False)

args = parser.parse_args()

DIM = args.dim
BATCH_SIZE = 1024
GPU_DEVICE = args.gpu
MAX_ITER = 30001
LR = 1e-3
SEED=int(datetime.now().timestamp() * 1000000) % 1000000
EMA_BETAS = [0.999, 0.99]
ema = EMA(0, betas=EMA_BETAS)
USE_WANDB = args.wandb

torch.cuda.set_device(GPU_DEVICE)

SAMPLER = args.sp # 'mb' or 'idp' or 'gt'
OT_MINIBATCH_SIZE = args.mb # for minibatch sampler

print("DIM: ", DIM)
print("GPU_DEVICE: ", GPU_DEVICE)
print("SAMPLER: ", SAMPLER)
print("OT_MINIBATCH_SIZE : ", OT_MINIBATCH_SIZE)

def seed_all(seed=123):
    OUTPUT_SEED = seed
    torch.manual_seed(OUTPUT_SEED)
    np.random.seed(OUTPUT_SEED)
    random.seed(OUTPUT_SEED)

seed_all(SEED)

import ofmsrc.map_benchmark as mbm
benchmark = mbm.Mix3ToMix10Benchmark(DIM)

def get_gt_plan_sample_fn(sampler_x, benchmark):
    
    def ret_fn(batch_size):
        x_samples = sampler_x.sample(batch_size)
        x_samples.requires_grad_()
        y_samples = benchmark.map_fwd(x_samples)
        return x_samples.clone().detach(), y_samples.clone().detach()
    
    return ret_fn

def get_indepedent_plan_sample_fn(sampler_x, sampler_y):
    
    def ret_fn(batch_size):
        x_samples = sampler_x.sample(batch_size)
        y_samples = sampler_y.sample(batch_size)
        return x_samples, y_samples
    
    return ret_fn


def get_discrete_ot_plan_sample_fn(sampler_x, sampler_y, device='cuda'):
    
    ot_plan_sampler = OTPlanSampler('exact')
    
    def ret_fn(batch_size):
        
        x_samples = sampler_x.sample(batch_size).to(device)
        y_samples = sampler_y.sample(batch_size).to(device)
        
        return ot_plan_sampler.sample_plan(x_samples, y_samples)
    
    return ret_fn

def get_discrete_smallbatch_ot_plan_sample_fn(sampler_x, sampler_y, mb_size=64, device='cuda'):
    sample_fn = get_discrete_ot_plan_sample_fn(sampler_x, sampler_y, device)

    def ret_fn(batch_size):
        spls_X = []
        spls_Y = []
        curr_sampled = 0
        while curr_sampled < batch_size:
            _X, _Y = sample_fn(mb_size)
            spls_X.append(_X); spls_Y.append(_Y)
            curr_sampled += mb_size
        X = torch.cat(spls_X); Y = torch.cat(spls_Y)
        return X[:batch_size], Y[:batch_size]

    return ret_fn

def PARAM2SAMPLER(sampler_type, mb_size=None):
    if sampler_type == 'idp':
        return get_indepedent_plan_sample_fn(
            benchmark.input_sampler, benchmark.output_sampler)
    if sampler_type == 'mb' and mb_size is None:
        return get_discrete_ot_plan_sample_fn(
            benchmark.input_sampler, benchmark.output_sampler)
    if sampler_type == 'mb' and isinstance(mb_size, int):
        return get_discrete_smallbatch_ot_plan_sample_fn(
            benchmark.input_sampler, benchmark.output_sampler, mb_size=mb_size)
    if sampler_type == 'gt':
        return get_gt_plan_sample_fn(benchmark.input_sampler, benchmark)
    raise Exception()


# SAMPLER = 'mb' # 'mb'
# OT_MINIBATCH_SIZE = 32
sampler_fn = PARAM2SAMPLER(SAMPLER, OT_MINIBATCH_SIZE)

D_HYPERPARAMS = {
    'dim' : DIM,
    'rank' : 1,
    'hidden_layer_sizes' : [max(2*DIM, 128), max(2*DIM, 128), max(DIM, 64)],
    'strong_convexity' : 1e-4
}


D = DenseICNN(**D_HYPERPARAMS).cuda()

pretrain_sampler = StandardNormalSampler(dim=DIM)
print('Pretraining identity potential. Final MSE:', train_identity_map(D, pretrain_sampler, convex=True, blow=3))
del pretrain_sampler

optim = torch.optim.RMSprop(D.parameters(), LR)

OUTPUT_PATH = '../logs/' + 'Mix3toMix10'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
CODE = ''.join(random.choices(string.ascii_lowercase, k=10))
CODE = f'dim_{DIM}_spl_{SAMPLER}_mb_{OT_MINIBATCH_SIZE}_' + CODE
RUN_PATH = os.path.join(OUTPUT_PATH, CODE)
os.makedirs(RUN_PATH)

print('TRAINING')
metrics = defaultdict(list)
L2_UVP_fwd_min = np.inf

WANDB_PROJECT_NAME = None # set your wandb project

if USE_WANDB:
    wandb_config = dict(
        dim=DIM,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        lr=LR,
        sampler=SAMPLER,
        ot_minibatch_size=OT_MINIBATCH_SIZE,
        seed=SEED,
        code=CODE
    )
    wandb.init(name=CODE, project=WANDB_PROJECT_NAME, reinit=True, config=wandb_config)

# inv params
LBFGS_TOLERANCE_GRAD = 5e-4 * np.sqrt(DIM)
LBFGS_MAX_ITER = 50

for iteration in range(MAX_ITER):
    current_metrics = dict()
    t_spl = time.perf_counter()
    X, Y = sampler_fn(BATCH_SIZE)
    current_metrics['times_spl'] = time.perf_counter() - t_spl
    
    t = (torch.rand(BATCH_SIZE) + 1e-8).cuda()

    loss_stats = dict()
    loss, true_loss = ofm_loss(D, X, Y, t, 
            TorchlbfgsInvParams(lbfgs_params=dict(tolerance_grad = LBFGS_TOLERANCE_GRAD), max_iter=LBFGS_MAX_ITER),
            BruteforceIHVParams(), 
            stats=loss_stats)
    
    optim.zero_grad()
    loss.backward()
    optim.step(); D.convexify();
    
    ema(D)
    
    # metrics
    current_metrics['loss'] = loss.item()
    current_metrics['true_loss'] = true_loss.item()
    for stat_type, stat_dict in loss_stats.items():
        if isinstance(stat_dict, dict):
            for k, v in stat_dict.items():
                mod_k = stat_type + '_' + k
                current_metrics[mod_k] = v
        elif isinstance(stat_dict, Number):
            current_metrics[stat_type] = stat_dict
        else:
            raise Exception('strange statistic')
    if (iteration % 25 == 1) and USE_WANDB:
        wandb.log(current_metrics, step=iteration)
    for k, v in current_metrics.items():
        metrics[k].append(v)
    
    if iteration % 100 == 1:
        L2_UVP_fwd, cos_fwd = score_fitted_map(benchmark, D)
        metrics['L2_UVP'].append(L2_UVP_fwd)
        metrics['cos'].append(cos_fwd)
        if USE_WANDB:
            wandb.log({
                'L2_UVP': L2_UVP_fwd,
                'cos': cos_fwd}, step=iteration)
        
        # model saving
        if L2_UVP_fwd < L2_UVP_fwd_min:
            torch.save(D.state_dict(), os.path.join(RUN_PATH, 'model.pth'))
            L2_UVP_fwd_min = L2_UVP_fwd
            for beta in EMA_BETAS:
                beta_model_path = os.path.join(RUN_PATH, 'model_{}.pth'.format(beta))
                torch.save(ema.get_model(beta).state_dict(), beta_model_path)
        
        torch.save(D.state_dict(), os.path.join(RUN_PATH, 'model_latest.pth'))
        for beta in EMA_BETAS:
            beta_model_path = os.path.join(RUN_PATH, 'model_latest_{}.pth'.format(beta))
            torch.save(ema.get_model(beta).state_dict(), beta_model_path)
        
        # statistics save
        statistics_save = dict(
            dim=DIM,
            batch_size=BATCH_SIZE,
            max_iter=MAX_ITER,
            lr=LR,
            sampler=SAMPLER,
            ot_minibatch_size=OT_MINIBATCH_SIZE,
            seed=SEED,
            code=CODE,
            metrics=metrics
        )

        with open(os.path.join(RUN_PATH, 'stats.pkl'), 'wb') as f:
            pickle.dump(statistics_save, f, protocol=pickle.HIGHEST_PROTOCOL)

if USE_WANDB:
    wandb.finish()