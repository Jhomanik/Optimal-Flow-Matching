import torch
from dataclasses import dataclass, field
from .icnn import GradNNGeneric
from typing import Optional
from .tools import freeze_context

###################
# PARAMETERS
############

@dataclass
class IHVMethodParamsGeneric:

    @property
    def name(self):
        raise NotImplementedError()

    verbose: bool = False

@dataclass
class BruteforceIHVParams(IHVMethodParamsGeneric):

    @property
    def name(self):
        return 'bruteforce'

    hess_batch_size: Optional[int] = None

###################
# SOLVERS
###########

def ofm_ihv_bruteforce(
        model: GradNNGeneric,
        z: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        method_params: BruteforceIHVParams
) -> torch.Tensor:
    assert len(z.shape) == 2 # (bs, dim)
    assert v.shape == z.shape

    hes = model.hessian_nograd(z, batch_size=method_params.hess_batch_size).squeeze(1) # (bs, dim, dim)
    assert len(hes.shape) == 3
    assert hes.size(0) == z.size(0)
    assert hes.size(1) == z.size(1)
    assert hes.size(2) == z.size(1)

    bias  = torch.eye(hes.shape[1]).repeat(z.size(0), 1, 1).to(z)
    assert t.size(0) == z.size(0)
    assert len(t.shape) == 1
    lhs = hes*t[:,None, None] + bias*(1-t[:, None, None])
    lhs = torch.linalg.inv(lhs).detach() # (bs, dim, dim)

    mvp = lhs.matmul(v.unsqueeze(-1)).squeeze(-1) # (bs, dim)

    return mvp

# main function

IHV_METHODS2SOLVERS = {
    'bruteforce': ofm_ihv_bruteforce
}

IHV_METHODS2PARAMCLASS = {
    'bruteforce': BruteforceIHVParams
}


def ofm_ihv(
        model: GradNNGeneric,
        z: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        method_params: IHVMethodParamsGeneric
) -> torch.Tensor:
    '''
    Calculates (Hess(model(z)) * t + I (1 - t))^{-1} v
    '''
    t = t.squeeze().to(z)
    assert len(t.shape) == 1
    assert t.size(0) == z.size(0)
    assert z.shape == v.shape
    assert method_params.name in IHV_METHODS2SOLVERS.keys()
    # assert isinstance(method_params, IHV_METHODS2PARAMCLASS[method_params.name])
    solver = IHV_METHODS2SOLVERS[method_params.name]
    with freeze_context(model):
        return solver(model, z, v, t, method_params)
