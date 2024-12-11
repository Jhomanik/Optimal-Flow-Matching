import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvexQuadratic(nn.Module):
    '''Convex Quadratic Layer'''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1,0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = F.linear(input, self.weight, self.bias)
        return quad + linear
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape)

class WeightTransformedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, w_transform=lambda x: x):
        super().__init__(in_features, out_features, bias=bias)
        self._w_transform = w_transform
    
    def forward(self, input):
        return F.linear(input, self._w_transform(self.weight), self.bias)

# from CPF
class ActNorm(torch.nn.Module):
    """ ActNorm layer with data-dependant init."""
    _scaling_min = 0.001

    def __init__(self, num_features, logscale_factor=1., scale=1., learn_scale=True, initialized=False):
        super(ActNorm, self).__init__()
        self.initialized = initialized
        self.num_features = num_features

        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))

    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            # noinspection PyShadowingNames
            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size)
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + self._scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

    def reverse(self, y, **kwargs):
        assert self.initialized
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        scale = torch.exp(logs) + self._scaling_min
        x = y / scale - b

        return x.view(input_shape)

    def extra_repr(self):
        return f"{self.num_features}"


class ActNormNoLogdet(ActNorm):

    def forward(self, x):
        return super(ActNormNoLogdet, self).forward_transform(x)[0]