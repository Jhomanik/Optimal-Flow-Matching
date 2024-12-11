import torch
import numpy as np
import random
from scipy.linalg import sqrtm
from sklearn import datasets
from .potentials import BasePotential

def symmetrize(X):
    return np.real((X + X.T) / 2)

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
    def _estimate_moments(self, size=2**14, mean=True, var=True, cov=True):
        if (not mean) and (not var) and (not cov):
            return

        sample = self.sample(size).cpu().detach().numpy().astype(np.float32)
        if mean:
            self.mean = sample.mean(axis=0)
        if var:
            self.var = sample.var(axis=0).sum()
        if cov:
            self.cov = np.cov(sample.T).astype(np.float32)
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)
    

class TensorSampler(Sampler):
    def __init__(self, tensor, device='cuda'):
        super(TensorSampler, self).__init__(device)
        self.tensor = torch.clone(tensor).to(device)
        
    def sample(self, size=5):
        assert size <= self.tensor.shape[0]
        
        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)


class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    
    
class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device='cuda'):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)

class NormalSampler(Sampler):
    def __init__(
        self, mean, cov=None, weight=None, device='cuda'
    ):
        super(NormalSampler, self).__init__(device=device)
        self.mean = np.array(mean, dtype=np.float32)
        self.dim = self.mean.shape[0]
        
        if weight is not None:
            weight = np.array(weight, dtype=np.float32)
        
        if cov is not None:
            self.cov = np.array(cov, dtype=np.float32)
        elif weight is not None:
            self.cov = weight @ weight.T
        else:
            self.cov = np.eye(self.dim, dtype=np.float32)
            
        if weight is None:
            weight = symmetrize(sqrtm(self.cov))
            
        self.var = np.trace(self.cov)
        
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        self.bias = torch.tensor(self.mean, device=self.device, dtype=torch.float32)

    def sample(self, size=4):
        sample = torch.randn(size, self.dim, device=self.device)
        with torch.no_grad():
            sample = sample @ self.weight.T
            if self.bias is not None:
                sample += self.bias
        return sample
    
    
class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    
    
class Mix8GaussiansSampler(Sampler):
    def __init__(self, with_central=False, std=1, r=12, dim=2, device='cuda'):
        super(Mix8GaussiansSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(batch_size, self.dim, device=self.device)
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        return batch

class Transformer(Sampler):
    def __init__(self, device='cuda'):
        self.device = device
        

class StandardNormalScaler(Transformer):
    def __init__(self, base_sampler, batch_size=1000, device='cuda'):
        super(StandardNormalScaler, self).__init__(device=device)
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()
        
        mean, cov = np.mean(batch, axis=0), np.cov(batch.T)
        
        self.mean = torch.tensor(
            mean, device=self.device, dtype=torch.float32
        )
        
        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=torch.float32
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier),
            device=self.device, dtype=torch.float32
        )
        torch.cuda.empty_cache()
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.tensor(self.base_sampler.sample(batch_size), device=self.device)
            batch -= self.mean
            batch @= self.inv_multiplier
        return batch
    
class LinearTransformer(Transformer):
    def __init__(
        self, base_sampler, weight, bias=None,
        device='cuda'
    ):
        super(LinearTransformer, self).__init__(device=device)
        self.base_sampler = base_sampler
        
        self.weight = torch.tensor(weight, device=device, dtype=torch.float32)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32)
        else:
            self.bias = torch.zeros(self.weight.size(0), device=device, dtype=torch.float32)
        
    def sample(self, size=4):        
        batch = torch.tensor(
            self.base_sampler.sample(size),
            device=self.device
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        return batch

class StandartNormalSampler(Sampler):
    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(StandartNormalSampler, self).__init__(
            device=device
        )
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(
            batch_size, self.dim, dtype=self.dtype,
            device=self.device, requires_grad=self.requires_grad
        )

class RandomGaussianMixSampler(Sampler):
    def __init__(
        self, dim=2, num=10, dist=1, std=0.4,
        standardized=True, estimate_size=2**14,
        batch_size=1024, device='cuda'
    ):
        super(RandomGaussianMixSampler, self).__init__(device=device)
        self.dim = dim
        self.num = num
        self.dist = dist
        self.std = std
        self.batch_size = batch_size
        
        centers = np.zeros((self.num, self.dim), dtype=np.float32)
        for d in range(self.dim):
            idx = np.random.choice(list(range(self.num)), self.num, replace=False)
            centers[:, d] += self.dist * idx
        centers -= self.dist * (self.num - 1) / 2
        
        maps = np.random.normal(size=(self.num, self.dim, self.dim)).astype(np.float32)
        maps /= np.sqrt((maps ** 2).sum(axis=2, keepdims=True))
        
        if standardized:
            mult = np.sqrt((centers ** 2).sum(axis=1).mean() + self.dim * self.std ** 2) / np.sqrt(self.dim)
            centers /= mult
            maps /= mult
        
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)  
        self.maps = torch.tensor(maps, device=self.device, dtype=torch.float32)
        
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self._estimate_moments(mean=False) # This can be also be done analytically
        
    def sample(self, size=10):          
        if size <= self.batch_size:
            idx = np.random.randint(0, self.num, size=size)
            sample = torch.randn(size, self.dim, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                sample = torch.matmul(self.maps[idx], sample[:, :, None])[:, :, 0] * self.std
                sample += self.centers[idx]
            return sample
        
        sample = torch.zeros(size, self.dim, dtype=torch.float32, device=self.device)
        for i in range(0, size, self.batch_size):
            batch = self.sample(min(i + self.batch_size, size) - i)
            with torch.no_grad():
                sample[i:i+self.batch_size] = batch
            torch.cuda.empty_cache()
        return sample
    
       
class PotentialTransformer(Transformer):
    def __init__(
        self, potential,
        device='cuda'
    ):
        super(PotentialTransformer, self).__init__(
            device=device
        )
        
        self.fitted = False
        
        assert issubclass(type(potential), BasePotential)
        self.potential = potential.to(self.device)
        self.dim = self.potential.dim
        
    def fit(self, base_sampler, estimate_size=2**14, estimate_cov=True):
        assert base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        self.fitted = True
        
        self._estimate_moments(estimate_size, True, True, estimate_cov)
        return self
        
    def sample(self, size=4):
        assert self.fitted == True
        sample = self.base_sampler.sample(size)
        sample.requires_grad_(True)
        return self.potential.push_nograd(sample)
    
class PushforwardTransformer(Transformer):
    def __init__(
        self, pushforward,
        batch_size=128,
        device='cuda'
    ):
        super(PushforwardTransformer, self).__init__(
            device=device
        )
        
        self.fitted = False
        self.batch_size = batch_size
        self.pushforward = pushforward

    def fit(self, base_sampler, estimate_size=2**14, estimate_cov=True):
        assert base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        self.fitted = True
        
        self._estimate_moments(estimate_size, True, True, estimate_cov)
        self.dim = len(self.mean)
        return self
        
    def sample(self, size=4):
        assert self.fitted == True
        
        if size <= self.batch_size:
            sample = self.base_sampler.sample(size)
            with torch.no_grad():
                sample = self.pushforward(sample)
            return sample
        
        sample = torch.zeros(size, self.sample(1).shape[1], dtype=torch.float32, device=self.device)
        for i in range(0, size, self.batch_size):
            batch = self.sample(min(i + self.batch_size, size) - i)
            with torch.no_grad():
                sample.data[i:i+self.batch_size] = batch.data
            torch.cuda.empty_cache()
        return sample
    
class StandardNormalScaler(Transformer):
    def __init__(self, device='cuda'):
        super(StandardNormalScaler, self).__init__(device=device)
        
    def fit(self, base_sampler, size=1000):
        assert self.base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        self.dim = self.base_sampler.dim
        
        self.bias = torch.tensor(
            self.base_sampler.mean, device=self.device, dtype=torch.float32
        )
        
        weight = symmetrize(np.linalg.inv(sqrtm(self.base_sampler.cov)))
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.cov = np.eye(self.dim, dtype=np.float32) # weight @ self.base_sampler.cov @ weight.T
        self.var = float(self.dim) #np.trace(self.cov)
        
        return self
        
    def sample(self, size=10):
        sample = torch.tensor(
            self.base_sampler.sample(size),
            device=self.device
        )
        with torch.no_grad():
            sample -= self.bias
            sample @= self.weight
        return sample
    
    
class NormalNoiseTransformer(Transformer):
    def __init__(
        self, std=0.01,
        device='cuda'
    ):
        super(NormalNoiseTransformer, self).__init__(
            device=device
        )
        self.std = std
        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        self.dim = base_sampler.dim
        self.mean = base_sampler.mean
        self.var = base_sampler.var + self.dim * (self.std ** 2)
        if hasattr(base_sampler, 'cov'):
            self.cov = base_sampler.cov + np.eye(self.dim, dtype=np.float32) * (self.std ** 2)
        return self
        
    def sample(self, batch_size=4):
        batch = self.base_sampler.sample(batch_size)
        with torch.no_grad():
            batch = batch + self.std * torch.randn_like(batch)
        return batch