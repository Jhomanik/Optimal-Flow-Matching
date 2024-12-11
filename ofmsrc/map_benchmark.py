import torch
import torch.nn as nn

import numpy as np
from scipy.linalg import sqrtm

from . import potentials
from . import distributions

from .bench_icnn import ConvICNN64, DenseICNN
from .bench_tools import freeze, load_resnet_G

import gc

def get_linear_transport_map(mean1, cov1, mean2, cov2):
    """Compute the linear optimal transport map weight matrix and bias vector between two distributions."""
    def symmetrize(X):
        return np.real((X + X.T) / 2)
    
    root_cov1 = symmetrize(sqrtm(cov1))
    inv_root_cov1 = symmetrize(np.linalg.inv(root_cov1))
    weight = inv_root_cov1 @ symmetrize(sqrtm(root_cov1 @ cov2 @ root_cov1)) @ inv_root_cov1
    bias = mean2 - weight @ mean1
    
    return weight, bias

def get_linear_transport_cost(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Computation of Bures-Wasserstein-2 metric. Based on https://github.com/mseitzer/pytorch-fid"""
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return .5 * (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

class Wasserstein2MapBenchmark:
    """Base class for all Wasserstein-2 map benchmarks."""
    def __init__(
        self, input_sampler, output_sampler,
        compute_linear=True, device='cuda'
    ):
        assert input_sampler.dim == output_sampler.dim
        assert input_sampler.device == output_sampler.device
        assert input_sampler.device == device
        
        self.input_sampler = input_sampler
        self.output_sampler = output_sampler
        self.dim =  input_sampler.dim
        
        self.device = device
        
        
        if compute_linear:
            self._compute_linear_transport()
            
        gc.collect(); torch.cuda.empty_cache()
        
    def _compute_linear_transport(self):
        weight, bias = get_linear_transport_map(
            self.input_sampler.mean, self.input_sampler.cov,
            self.output_sampler.mean, self.output_sampler.cov,
        )
        map_fwd = nn.Linear(self.dim, self.dim).to(self.device)
        map_fwd.weight.data = torch.tensor(weight, device=self.device, dtype=torch.float32)
        map_fwd.bias.data = torch.tensor(bias, device=self.device, dtype=torch.float32)
        self.linear_map_fwd = map_fwd
        
        weight_inv, bias_inv = get_linear_transport_map(
            self.output_sampler.mean, self.output_sampler.cov,
            self.input_sampler.mean, self.input_sampler.cov,
        )
        map_inv = nn.Linear(self.dim, self.dim).to(self.device)
        map_inv.weight.data = torch.tensor(weight_inv, device=self.device, dtype=torch.float32)
        map_inv.bias.data = torch.tensor(bias_inv, device=self.device, dtype=torch.float32)
        self.linear_map_inv = map_inv
        
        self.linear_cost = get_linear_transport_cost(
            self.output_sampler.mean, self.output_sampler.cov,
            self.input_sampler.mean, self.input_sampler.cov
        )

class PotentialMapBenchmark(Wasserstein2MapBenchmark):
    def __init__(
        self, input_sampler, potential,
        compute_linear=True,
        estimate_size=2**14,
        estimate_cov=True,
        batch_size=1024,
        device='cuda'
    ):
        assert input_sampler.device == device
        self.potential = potential
        output_sampler = distributions.PotentialTransformer(
            potential, device=device,
        ).fit(
            input_sampler,
            estimate_size=estimate_size, 
            estimate_cov=estimate_cov,
        )
        
        if not estimate_cov:
            assert compute_linear == False
        
        super(PotentialMapBenchmark, self).__init__(
            input_sampler, output_sampler,
            compute_linear=compute_linear, device=device
        )
        
        self.input_sampler = input_sampler
        self.dim = input_sampler.dim
        self.batch_size = batch_size
        self.device = device
        
        self._estimate_cost(estimate_size)
        
    def map_fwd(self, input, nograd=True):
        if nograd:
            return self.potential.push_nograd(input)
        return self.potential.push(input)
    
    def _estimate_cost(self, estimate_size):
        X = self.input_sampler.sample(self.batch_size)
        X.requires_grad_(True)
        X_push = self.map_fwd(X, nograd=True)
        with torch.no_grad():
            self.cost = .5 * ((X - X_push) ** 2).sum(dim=1).mean(dim=0).item()
        return self.cost
    

class CelebA64Benchmark(PotentialMapBenchmark):
    def __init__(
        self,
        which='Early',
        batch_size=64,
        device='cuda'
    ):
        assert which in ["Early", "Mid", "Late"]
        
        # Load ResNet and Create Input Sampler
        resnet = load_resnet_G('../benchmarks/CelebA64/Final_G.pt')
        input_sampler = distributions.NormalNoiseTransformer(std=0.01).fit(
            distributions.PushforwardTransformer(resnet).fit(
                distributions.StandardNormalSampler(dim=128), estimate_cov=False
            )
        )
        freeze(resnet)
        gc.collect(); torch.cuda.empty_cache()
        
        # Load first potential and freeze
        D1 = ConvICNN64().to(device); 
        D1.load_state_dict(
            torch.load(
                f'../benchmarks/CelebA64/{which}_v1.pt',
                map_location=lambda storage, loc: storage
            )
        )
        freeze(D1)
        
        # Load second potential and freeze
        D2 = ConvICNN64().to(device);
        D2.load_state_dict(
            torch.load(
                f'../benchmarks/CelebA64/{which}_v2.pt',
                map_location=lambda storage, loc: storage
            )
        )
        freeze(D2)
        
        # Mix potentials
        potential = .5 * (potentials.Potential(D1, batch_size=2*batch_size) + potentials.Potential(D2, batch_size=2*batch_size))
        
        super(CelebA64Benchmark, self).__init__(
            input_sampler, potential,
            compute_linear=False,
            estimate_size=2**14,
            estimate_cov=False,
            batch_size=batch_size,
            device=device
        )     
        
        gc.collect(); torch.cuda.empty_cache()
        
class Mix3ToMix10Benchmark(PotentialMapBenchmark):
    def __init__(self, dim=2, batch_size=1024, device='cuda'):
        assert dim in [2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        np.random.seed(0x000000); torch.manual_seed(0x000000)
        input_sampler = distributions.RandomGaussianMixSampler(dim=dim, num=3)
        
        D1 = DenseICNN(
            dim=dim, rank=1, strong_convexity=1e-2,
            hidden_layer_sizes=[max(2*dim, 64), max(2*dim, 64), max(dim, 32)]
        ).to(device)
        D1.load_state_dict(
            torch.load(
                f'../benchmarks/Mix3toMix10/{dim}_v1.pt',
                map_location=lambda storage, loc: storage
            )
        )
        freeze(D1)
        
        D2 = DenseICNN(
            dim=dim, rank=1, strong_convexity=1e-2,
            hidden_layer_sizes=[max(2*dim, 64), max(2*dim, 64), max(dim, 32)]
        ).to(device)
        D2.load_state_dict(
            torch.load(
                f'../benchmarks/Mix3toMix10/{dim}_v2.pt',
                map_location=lambda storage, loc: storage
            )
        ) 
        freeze(D2)
        
        # Mix potentials
        potential = potentials.Potential(D1, batch_size=2*batch_size) + potentials.Potential(D2, batch_size=2*batch_size)
        potential = potentials.standardize_potential(potential, input_sampler)
        
        super(Mix3ToMix10Benchmark, self).__init__(
            input_sampler, potential,
            batch_size=batch_size,
            device=device
        )     
        
        gc.collect(); torch.cuda.empty_cache()