import torch
import torch.distributions as TD
import scipy.stats as sps
from typing import Optional

class OOD:

    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.size(0), dtype=torch.bool).to(x).bool()

class OODMahalanobis(OOD):
    
    @staticmethod
    def chi2(prob, dim):
        return sps.chi2.ppf(prob, dim)

    def __init__(self, X: torch.Tensor, m: Optional[float] = None):
        assert len(X.shape) == 2
        self.dim = X.size(1)
        self.m = self.chi2(0.95, self.dim) * 2. if m is None else m
        self.mean = torch.mean(X, dim=0)
        assert self.mean.size(0) == self.dim
        self.var = torch.cov(X.T)
        self.var_inverse = torch.linalg.inv(self.var)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2
        assert x.size(1) == self.dim
        diff = x - self.mean[None,:]
        m_obt = torch.sum(diff * torch.matmul(self.var_inverse, diff.T).T, axis=1)
        return m_obt < self.m
        