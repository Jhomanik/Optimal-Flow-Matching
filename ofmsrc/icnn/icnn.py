import torch
import torch.nn as nn

from .layers import (
    WeightTransformedLinear,
    ActNormNoLogdet
)
from .icnn_cpf import (
    GradICNNGenericCPF,
    Softplus,
    SymmSoftplus,
    symm_softplus
)

class LinActnormICNN(GradICNNGenericCPF):

    def __init__(
            self, 
            dim=2, 
            dimh=16, 
            num_hidden_layers=2, 
            symm_act_first=False,
            softplus_type='softplus', 
            zero_softplus=False, 
            batch_size=1024, 
            actnorm_initialized=False,
            conv_layers_w_trf=lambda x: x,
            forse_w_positive=True,
        ):
        super().__init__(batch_size=batch_size)
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first
        self.conv_layers_w_trf = conv_layers_w_trf
        self.forse_w_positive = forse_w_positive

        Wzs = list()
        
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(
                WeightTransformedLinear(
                    dimh, dimh, bias=True, w_transform=self.conv_layers_w_trf))
        Wzs.append(WeightTransformedLinear(
                    dimh, 1, bias=False, w_transform=self.conv_layers_w_trf))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        self.actnorm0 = ActNormNoLogdet(dimh, initialized=actnorm_initialized)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh, initialized=actnorm_initialized))
        actnorms.append(ActNormNoLogdet(1, initialized=actnorm_initialized))
        self.actnorms = torch.nn.ModuleList(actnorms)
        self.convexify()

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        for Wz, Wx, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + Wx(x)))
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))
    
    def convexify(self):
        if self.forse_w_positive:
            for layer in self.Wzs:
                if (isinstance(layer, WeightTransformedLinear)):
                    layer.weight.data.clamp_(0)