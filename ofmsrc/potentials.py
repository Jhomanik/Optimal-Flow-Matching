import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class BasePotential(nn.Module):
    def __init__(self, batch_size=1024):
        super(BasePotential, self).__init__()
        self.batch_size = batch_size
    
    def __radd__(self, other):
        return WeightedSumPotential([self, other], [1, 1], batch_size=max(1, (self.batch_size + other.batch_size) // 4))
    
    def __add__(self, other):
        return WeightedSumPotential([self, other], [1, 1], batch_size=max(1, (self.batch_size + other.batch_size) // 4))
    
    def __rmul__(self, other):
        assert other >= 0
        return ScaledPotential(self, other, batch_size=self.batch_size)
    
    def __mul__(self, other):
        assert other >= 0
        return ScaledPotential(self, other, batch_size=self.batch_size)
    
    def forward(self, input):
        pass
      
    def push(self, input, create_graph=True, retain_graph=True):
        assert len(input) <= self.batch_size
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=create_graph, retain_graph=retain_graph,
            only_inputs=True,
            grad_outputs=torch.ones_like(input[:, [0]])
        )[0]
        return output
    
    def push_nograd(self, input):
        '''
        Pushes input by using the gradient of the network. Does not preserve the computational graph.
        Use for pushing large batches (the function uses minibatches).
        '''
        output = torch.zeros_like(input, requires_grad=False)
        for i in range(0, len(input), self.batch_size):
            input_batch = input[i:i+self.batch_size]
            output.data[i:i+self.batch_size] = self.push(
                input[i:i+self.batch_size],
                create_graph=False, retain_graph=False
            ).data
        return output 
    
class Potential(BasePotential):
    def __init__(self, potential, batch_size=1024):
        super(Potential, self).__init__(batch_size)
        self.dim = potential.dim
        self.potential = potential
        
    def forward(self, input):
        return self.potential(input)
    
class ScaledPotential(BasePotential):
    def __init__(self, potential, scale, batch_size=1024):
        super(ScaledPotential, self).__init__(batch_size)
        assert isinstance(potential, BasePotential)
        assert scale >= 0
            
        self.dim = potential.dim
        self.potential = potential
        self.scale = scale
        
    def forward(self, input):
        return self.scale * self.potential(input)
    
class ShiftedPotential(BasePotential):
    def __init__(self, potential, shift, batch_size=1024):
        super(ShiftedPotential, self).__init__(batch_size)
        assert issubclass(type(potential), BasePotential)
            
        self.dim = potential.dim
        assert len(shift) == self.dim
        self.potential = potential
        self.shift = torch.tensor(shift, dtype=torch.float32, device='cuda')
        
    def forward(self, input):
        return self.potential(input) + (self.shift * input).sum(dim=1, keepdims=True)
    
class WeightedSumPotential(BasePotential):
    def __init__(self, potentials, weights, batch_size=1024):
        super(WeightedSumPotential, self).__init__(batch_size)
        assert len(potentials) > 0
        assert len(potentials) == len(weights)
        for weight, potential in zip(weights, potentials):
            assert issubclass(type(potential), BasePotential)
            assert potential.dim == potentials[0].dim
            assert weight >= 0
            
        self.dim = potentials[0].dim
        self.potentials = potentials
        self.weights = weights
        
    def forward(self, input):
        output = 0.
        for weight, potential in zip(self.weights, self.potentials):
            output += potential(input) * weight
        return output
    
def standardize_potential(potential, sampler, size=2**14):
    """
    Outputs the linearly scaled potential, i.e. [a*potential+b*x].
    This scaled output potential pushes sampler's distributions
    to zero-mean distribution with variance equal to the potential.dim.
    """
    X = sampler.sample(size); X.requires_grad_(True)
    Y = potential.push_nograd(X).cpu().numpy()
    mean = np.mean(Y, axis=0)
    mean_var = (Y - mean).var(axis=0).mean()
    
    return ScaledPotential(ShiftedPotential(potential, -mean), 1. / np.sqrt(mean_var), batch_size=potential.batch_size)