import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvexQuadratic, WeightTransformedLinear
from .gradnn import GradICNNGeneric


class DenseICNN(GradICNNGeneric):
    '''Fully Conncted ICNN with input-quadratic skip connections.'''

    def __init__(
        self, dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu',
        strong_convexity=1e-6,
        batch_size=1024,
        conv_layers_w_trf=lambda x: x,
        forse_w_positive=True,
        weights_init_std=0.1
    ):
        super().__init__(batch_size)
        
        self.dim = dim
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.conv_layers_w_trf = conv_layers_w_trf
        self.forse_w_positive = forse_w_positive
        self.weights_init_std = weights_init_std
        
        self.quadratic_layers = nn.ModuleList([
            ConvexQuadratic(dim, out_features, rank=rank, bias=True)
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            WeightTransformedLinear(
                in_features, out_features, bias=False, w_transform=self.conv_layers_w_trf)
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = WeightTransformedLinear(
            hidden_layer_sizes[-1], 1, bias=False, w_transform=self.conv_layers_w_trf)
        
        if self.weights_init_std is not None:
            self._init_weights(self.weights_init_std)
        self.convexify()

    def _init_weights(self, std):
        for p in self.parameters():
            p.data = (torch.randn(p.shape, dtype=torch.float32) * std).to(p)  

    def forward(self, input):
        '''Evaluation of the discriminator value. Preserves the computational graph.'''
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def convexify(self):
        if self.forse_w_positive:
            for layer in self.convex_layers:
                if (isinstance(layer, nn.Linear)):
                    layer.weight.data.clamp_(0)
            self.final_layer.weight.data.clamp_(0)