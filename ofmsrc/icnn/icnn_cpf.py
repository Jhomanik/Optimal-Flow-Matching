import torch
# noinspection PyPep8Naming
from torch import nn, Tensor
import torch.nn.init as init
# noinspection PyPep8Naming
import torch.nn.functional as F
import numpy as np
from .functional import log_sum_exp
from .gradnn import GradICNNGeneric
from .layers import ActNormNoLogdet


def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


def softplus(x):
    return nn.functional.softplus(x)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / (2*z)


def gaussian_softplus2(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / z


def laplace_softplus(x):
    return torch.relu(x) + torch.exp(-torch.abs(x)) / 2


def cauchy_softplus(x):
    # (Pi y + 2 y ArcTan[y] - Log[1 + y ^ 2]) / (2 Pi)
    pi = np.pi
    return (x * pi - torch.log(x**2 + 1) + 2 * x * torch.atan(x)) / (2*pi)


def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))
    return shifted_activation


def get_softplus(softplus_type='softplus', zero_softplus=False):
    if softplus_type == 'softplus':
        act = nn.functional.softplus
    elif softplus_type == 'gaussian_softplus':
        act = gaussian_softplus
    elif softplus_type == 'gaussian_softplus2':
        act = gaussian_softplus2
    elif softplus_type == 'laplace_softplus':
        act = gaussian_softplus
    elif softplus_type == 'cauchy_softplus':
        act = cauchy_softplus
    else:
        raise NotImplementedError(f'softplus type {softplus_type} not supported.')
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type='softplus', zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class SymmSoftplus(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return symm_softplus(x)


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return nn.functional.linear(x, torch.nn.functional.softplus(self.weight), self.bias) * gain


class PosLinear2(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, torch.nn.functional.softmax(self.weight, 1), self.bias)


class PosConv2d(torch.nn.Conv2d):

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # noinspection PyProtectedMember,PyAttributeOutsideInit
        self.fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, torch.nn.functional.softplus(self.weight), None) / self.fan_in

class GradICNNGenericCPF(GradICNNGeneric):

    def convexify(self):
        pass


# noinspection PyPep8Naming,PyTypeChecker
class ICNNCPF(GradICNNGenericCPF):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=1, batch_size=1024):
        super().__init__(batch_size=batch_size)

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh, bias=False))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)

# noinspection PyPep8Naming,PyTypeChecker
class ICNN2CPFnoSPact(GradICNNGenericCPF):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=2, symm_act_first=False,
                 act_type='celu', batch_size=1024, 
                 actnorm_initialized=False):
        super().__init__(batch_size=batch_size)
        # with data dependent init

        if act_type == 'celu':
            self.act = nn.CELU()
        elif act_type == 'lrelu':
            self.act = nn.LeakyReLU()
        else:
            raise Exception(f'unsupported act type "{act_type}"')
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
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

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        for Wz, Wx, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + Wx(x)))
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))

# noinspection PyPep8Naming,PyTypeChecker
class ICNN2CPF(GradICNNGenericCPF):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=2, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False, batch_size=1024, 
                 actnorm_initialized=False):
        super().__init__(batch_size=batch_size)
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
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

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        for Wz, Wx, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + Wx(x)))
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))


# noinspection PyPep8Naming,PyTypeChecker
class ICNN3CPF(GradICNNGenericCPF):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=2, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False, batch_size=1024, 
                 actnorm_initialized=False):
        super().__init__(batch_size=batch_size)
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh // 2, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh // 2, initialized=actnorm_initialized))
        actnorms.append(ActNormNoLogdet(1, initialized=actnorm_initialized))
        actnorms[-1].b.requires_grad_(False)
        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.Wzs[0](x), self.act)
        else:
            z = self.act(self.Wzs[0](x))
        for Wz, Wx, Wx2, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s[:], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + Wx(x)))
            aug = Wx2(x)
            aug = symm_softplus(aug, self.act) if self.symm_act_first else self.act(aug)
            z = torch.cat([z, aug], 1)
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))


# noinspection PyPep8Naming,PyUnusedLocal
class LseICNNCPF(GradICNNGenericCPF):
    def __init__(self, dim=2, dimh=16, batch_size=1024, **kargs):
        super().__init__(batch_size=batch_size)
        self.L = torch.nn.Linear(dim, dimh)

    def forward(self, x):
        return log_sum_exp(self.L(x), -1)


# noinspection PyPep8Naming,PyTypeChecker
class ResICNN2CPF(GradICNNGenericCPF):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=2, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False, batch_size=1024,
                 actnorm_initialized=False):
        super().__init__(batch_size=batch_size)
        # with data dependent init

        assert num_hidden_layers > 1, 'num_hidden_layers <= 1'
        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(nn.Sequential(
                PosLinear(dimh, dimh, bias=False),
                ActNormNoLogdet(dimh, initialized=actnorm_initialized),
                nn.Softplus(),
                PosLinear(dimh, dimh, bias=True)
            ))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh, bias=False))  # not needed cos of actnorm
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        self.actnorm0 = ActNormNoLogdet(dimh, initialized=actnorm_initialized)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh, initialized=actnorm_initialized))
        actnorms.append(ActNormNoLogdet(1, initialized=actnorm_initialized))

        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        for Wz, Wx, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + z + Wx(x)))
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))


# noinspection PyPep8Naming,PyTypeChecker
class DenseICNN2CPF(GradICNNGenericCPF):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=2, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False, batch_size=1024):
        super().__init__(batch_size=batch_size)
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for j in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh * (j + 1), dimh, bias=False))
        Wzs.append(PosLinear(dimh * num_hidden_layers, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh, bias=False))  # not needed cos of actnorm
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        self.actnorm0 = ActNormNoLogdet(dimh)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh))
        actnorms.append(ActNormNoLogdet(1))

        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        for Wz, Wx, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.actnorms[:-1]):
            z_ = self.act(actnorm(Wz(z) + Wx(x)))
            z = torch.cat([z, z_], 1)
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))


# noinspection PyPep8Naming,PyTypeChecker
class ConvICNNCPF(GradICNNGenericCPF):

    def __init__(self, dim=1, dimh=16, num_hidden_layers=2, num_pooling=0, batch_size=1024):
        in_channels = dim
        hid_channels = dimh

        # total num of forward convolutions: num_hidden_layers * (num_max_pooling + 1) * 2, 1 for z one for x

        super().__init__(batch_size=batch_size)
        # with data dependent init

        # assert num_hidden_layers > 1, 'num_hidden_layers <= 1'
        self.act = nn.Softplus()

        self.Wz0 = nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.Wx0 = nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1)

        steps = list()

        # FIRST STEP
        step = list()
        Wzs = list()
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosConv2d(hid_channels, hid_channels, kernel_size=3, padding=1, bias=True))
        step.append(torch.nn.ModuleList(Wzs))

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Conv2d(hid_channels, hid_channels, kernel_size=3, padding=1))
        step.append(torch.nn.ModuleList(Wxs))

        actnorms = list()
        for _ in range(num_hidden_layers):
            actnorms.append(ActNormNoLogdet(hid_channels))
        step.append(torch.nn.ModuleList(actnorms))

        steps.append(torch.nn.ModuleList(step))

        self.num_pooling = num_pooling
        self.pooling_layers_z = nn.ModuleList([
            PosConv2d(hid_channels, hid_channels, kernel_size=2, stride=2, padding=2, bias=False)
            for _ in range(self.num_pooling)
        ])
        self.pooling_layers_x = nn.ModuleList([
            nn.Conv2d(hid_channels, hid_channels, kernel_size=2, stride=2, padding=2, bias=False)
            for _ in range(self.num_pooling)
        ])
        for s in range(num_pooling):
            step = list()
            Wzs = list()
            for _ in range(num_hidden_layers):
                Wzs.append(PosConv2d(hid_channels, hid_channels, kernel_size=3, padding=1, bias=True))
            step.append(torch.nn.ModuleList(Wzs))

            Wxs = list()
            for _ in range(num_hidden_layers):
                Wxs.append(nn.Conv2d(hid_channels, hid_channels, kernel_size=3, padding=1))
            step.append(torch.nn.ModuleList(Wxs))

            actnorms = list()
            for _ in range(num_hidden_layers):
                actnorms.append(ActNormNoLogdet(hid_channels))
            step.append(torch.nn.ModuleList(actnorms))

            steps.append(torch.nn.ModuleList(step))

        self.steps = torch.nn.ModuleList(steps)
        self.actnorm_first = ActNormNoLogdet(hid_channels)
        # self.actnorm_last = ActNormNoLogdet(1)

    def forward(self, x):
        z = self.act(self.actnorm_first(self.Wz0(x)))
        x = self.Wx0(x)

        for s, step in enumerate(self.steps):
            Wzs, Wxs, actnorms = step[0], step[1], step[2]
            for Wz, Wx, actnorm in zip(Wzs, Wxs, actnorms):
                z = self.act(actnorm((Wz(z) + Wx(x))))
            if s != len(self.steps) - 1:
                x = self.pooling_layers_x[s](x)
                z = self.pooling_layers_z[s](z)
        return nn.functional.adaptive_avg_pool2d(z + x, 1).squeeze().sum(1, keepdim=True)


# noinspection PyPep8Naming,PyTypeChecker
class ConvICNN2CPF(GradICNNGenericCPF):

    def __init__(self, dim=1, dimh=16, num_hidden_layers=2, symm_act_first=True, 
                 softplus_type="gaussian_softplus2", zero_softplus=True, batch_size=1024):
        super().__init__(batch_size=batch_size)
        in_channels = dim
        hid_channels = dimh

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        def conv(d_in, d_out, bias=True):
            return nn.Conv2d(d_in, d_out, kernel_size=3, padding=1, bias=bias)

        def pos_conv(d_in, d_out, bias=True):
            return PosConv2d(d_in, d_out, kernel_size=3, padding=1, bias=bias)

        self.act = nn.Softplus()

        self.Wz0 = conv(in_channels, hid_channels)
        self.Wx0 = conv(in_channels, hid_channels)

        # FIRST STEP
        step = list()
        Wzs = list()
        for _ in range(num_hidden_layers - 1):
            Wzs.append(pos_conv(hid_channels, hid_channels // 2))
        step.append(torch.nn.ModuleList(Wzs))

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(conv(hid_channels, hid_channels // 2))
        step.append(torch.nn.ModuleList(Wxs))

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(conv(hid_channels, hid_channels // 2))
        Wx2s = torch.nn.ModuleList(Wx2s)
        step.append(torch.nn.ModuleList(Wx2s))

        actnorms = list()
        for _ in range(num_hidden_layers):
            actnorms.append(ActNormNoLogdet(hid_channels // 2))
        step.append(torch.nn.ModuleList(actnorms))

        self.step = torch.nn.ModuleList(step)
        self.actnorm_first = ActNormNoLogdet(hid_channels)

    def forward(self, x):
        z = self.act(self.actnorm_first(self.Wz0(x)))
        x = self.Wx0(x)

        for Wz, Wx, Wx2, actnorm in zip(*self.step):
            z = self.act(actnorm((Wz(z) + Wx(x))))
            z = torch.cat([z, symm_softplus(Wx2(x))], 1)
        return nn.functional.adaptive_avg_pool2d(z, 1).squeeze().sum(1, keepdim=True)


# noinspection PyPep8Naming,PyTypeChecker
class ConvICNN3CPF(GradICNNGenericCPF):

    def __init__(self, dim=1, dimh=16, num_hidden_layers=2,
                 symm_act_first=True, softplus_type="softplus",
                 zero_softplus=True, batch_size=1024):
        super().__init__(batch_size=batch_size)

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        def conv(d_in, d_out, bias=True):
            return nn.Conv2d(d_in, d_out, kernel_size=3, padding=1, bias=bias)

        def pos_conv(d_in, d_out, bias=True):
            return PosConv2d(d_in, d_out, kernel_size=3, padding=1, bias=bias)

        Wzs = list()
        Wzs.append(conv(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(pos_conv(dimh, dimh // 2, True))
        Wzs.append(pos_conv(dimh, 1, False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(conv(dim, dimh // 2))
        Wxs.append(conv(dim, 1, False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(conv(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

        actnorms = list()
        for _ in range(num_hidden_layers):
            actnorms.append(ActNormNoLogdet(dimh // 2))
        actnorms.append(ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x):
        z = symm_softplus(self.Wzs[0](x), self.act)
        for Wz, Wx, Wx2, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s[:], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + Wx(x)))
            z = torch.cat([z, symm_softplus(Wx2(x))], 1)
        out = self.Wzs[-1](z) + self.Wxs[-1](x)
        out = nn.functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        return out


# noinspection PyPep8Naming
class DenseNetICNN(nn.Module):
    r"""Based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
    """

    def __init__(self, dim=1, dimh=16, num_hidden_layers=2, num_pooling=0, bn_size=4, num_init_features=None):
        raise NotImplementedError()
        super(DenseNetICNN, self).__init__()

        if num_init_features is None:
            num_init_features = dimh

        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(dim, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            ActNormNoLogdet(num_init_features),
            SymmSoftplus(),
        )

        # Each denseblock
        num_features = num_init_features
        for i in range(num_pooling + 1):
            block = _DenseBlock(
                num_layers=num_hidden_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=dimh,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_hidden_layers * dimh
            if i != num_pooling:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', ActNormNoLogdet(num_features))

        # Linear layer
        self.fc_final = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc_final(out)
        return out


# noinspection PyPep8Naming
class _DenseLayer(torch.nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size=4):
        super(_DenseLayer, self).__init__()
        self.norm1 = ActNormNoLogdet(num_input_features)
        self.act1 = nn.Softplus()
        self.conv1 = PosConv2d(num_input_features, bn_size *
                               growth_rate, kernel_size=1, stride=1,
                               bias=False)
        self.norm2 = ActNormNoLogdet(bn_size * growth_rate)
        self.act2 = nn.Softplus()
        self.conv2 = PosConv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1,
                               bias=False)

    def forward(self, inputs):
        if isinstance(inputs, Tensor):
            inputs = [inputs]

        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.act1(self.norm1(concated_features)))
        new_features = self.conv2(self.act2(self.norm2(bottleneck_output)))
        return new_features


# noinspection PyPep8Naming
class _DenseBlock(torch.nn.Module):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', ActNormNoLogdet(num_input_features))
        self.add_module('act', nn.Softplus())
        self.add_module('conv', PosConv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class PICNNAbstractClass(torch.nn.Module):
    icnns = dict()
    icnn_names = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.icnns[cls.__name__] = cls
        cls.icnn_names.append(cls.__name__)


# noinspection PyPep8Naming,PyTypeChecker
class PICNN(PICNNAbstractClass):
    def __init__(self, dim=2, dimh=16, dimc=2, num_hidden_layers=2, PosLin=PosLinear,
                 symm_act_first=False, softplus_type='gaussian_softplus', zero_softplus=False):
        super(PICNN, self).__init__()
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first

        # data path
        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh, dimh, bias=True))
        Wzs.append(PosLin(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        # context path
        Wcs = list()
        Wcs.append(nn.Linear(dimc, dimh))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = list()
        for _ in range(num_hidden_layers - 1):
            Wczs.append(nn.Linear(dimh, dimh))
        Wczs.append(nn.Linear(dimh, dimh, bias=True))
        self.Wczs = torch.nn.ModuleList(Wczs)
        for Wcz in self.Wczs:
            Wcz.weight.data.zero_()
            Wcz.bias.data.zero_()

        Wcxs = list()
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(nn.Linear(dimh, dim))
        Wcxs.append(nn.Linear(dimh, dim, bias=True))
        self.Wcxs = torch.nn.ModuleList(Wcxs)
        for Wcx in self.Wcxs:
            Wcx.weight.data.zero_()
            Wcx.bias.data.zero_()

        Wccs = list()
        for _ in range(num_hidden_layers - 1):
            Wccs.append(nn.Linear(dimh, dimh))
        self.Wccs = torch.nn.ModuleList(Wccs)

        self.actnorm0 = ActNormNoLogdet(dimh)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh))
        actnorms.append(ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

        self.actnormc = ActNormNoLogdet(dimh)

    def forward(self, x, c):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        c = self.act_c(self.actnormc(self.Wcs[0](c)))
        for Wz, Wx, Wcz, Wcx, Wcc, actnorm in zip(
                self.Wzs[1:-1], self.Wxs[:-1],
                self.Wczs[:-1], self.Wcxs[:-1], self.Wccs,
                self.actnorms[:-1]):
            cz = softplus(Wcz(c) + np.exp(np.log(1.0) - 1))
            cx = Wcx(c) + 1.0
            z = self.act(actnorm(Wz(z * cz) + Wx(x * cx) + Wcc(c)))

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.actnorms[-1](
            self.Wzs[-1](z * cz) + self.Wxs[-1](x * cx)
        )


# noinspection PyPep8Naming,PyTypeChecker
class PICNN2(PICNNAbstractClass):
    def __init__(self, dim=2, dimh=16, dimc=2, num_hidden_layers=2, PosLin=PosLinear, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False):
        super(PICNN2, self).__init__()
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first

        # data path
        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh, dimh // 2, bias=True))
        Wzs.append(PosLin(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

        # context path
        Wcs = list()
        Wcs.append(nn.Linear(dimc, dimh))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = list()
        for _ in range(num_hidden_layers - 1):
            Wczs.append(nn.Linear(dimh, dimh))
        Wczs.append(nn.Linear(dimh, dimh, bias=True))
        self.Wczs = torch.nn.ModuleList(Wczs)
        for Wcz in self.Wczs:
            Wcz.weight.data.zero_()
            Wcz.bias.data.zero_()

        Wcxs = list()
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(nn.Linear(dimh, dim))
        Wcxs.append(nn.Linear(dimh, dim, bias=True))
        self.Wcxs = torch.nn.ModuleList(Wcxs)
        for Wcx in self.Wcxs:
            Wcx.weight.data.zero_()
            Wcx.bias.data.zero_()

        Wcx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wcx2s.append(nn.Linear(dimh, dim))
        self.Wcx2s = torch.nn.ModuleList(Wcx2s)
        for Wcx2 in self.Wcx2s:
            Wcx2.weight.data.zero_()
            Wcx2.bias.data.zero_()

        Wccs = list()
        for _ in range(num_hidden_layers - 1):
            Wccs.append(nn.Linear(dimh, dimh))
        self.Wccs = torch.nn.ModuleList(Wccs)

        self.actnorm0 = ActNormNoLogdet(dimh)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh // 2))
        actnorms.append(ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

        actnorm2s = list()
        for _ in range(num_hidden_layers - 1):
            actnorm2s.append(ActNormNoLogdet(dimh // 2))
        self.actnorm2s = torch.nn.ModuleList(actnorm2s)

        self.actnormc = ActNormNoLogdet(dimh)

    def forward(self, x, c):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
            # z = symm_softplus((self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
            # z = self.act((self.Wzs[0](x)))
        # c = self.act_c(self.actnormc(self.Wcs[0](c)))
        c = self.act_c((self.Wcs[0](c)))
        for Wz, Wx, Wx2, Wcz, Wcx, Wcx2, Wcc, actnorm, actnorm2 in zip(
                self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s,
                self.Wczs[:-1], self.Wcxs[:-1], self.Wcx2s, self.Wccs,
                self.actnorms[:-1], self.actnorm2s):
            cz = softplus((Wcz(c) + np.exp(np.log(1.0) - 1)))
            cx = Wcx(c) + 1.0
            cx2 = Wcx2(c) + 1.0
            c1, c2 = torch.chunk(Wcc(c), 2, dim=1)
            z = self.act(actnorm(Wz(z * cz) + Wx(x * cx) + c1))
            aug = Wx2(x * cx2) + c2
            # aug = actnorm2(aug)
            aug = symm_softplus(aug, self.act) if self.symm_act_first else self.act(aug)
            z = torch.cat([z, aug], 1)

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.actnorms[-1](
            self.Wzs[-1](z * cz) + self.Wxs[-1](x * cx)
        )


# noinspection PyPep8Naming,PyTypeChecker
class PICNNFW(PICNNAbstractClass):
    """
    feature-wise transformation conditioning (https://distill.pub/2018/feature-wise-transformations/)
    """
    def __init__(self, dim=2, dimh=16, dimc=2, num_hidden_layers=2, PosLin=PosLinear, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False):
        super(PICNNFW, self).__init__()
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first

        # data path
        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh, dimh // 2, bias=True))
        Wzs.append(PosLin(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

        # context path
        Wcs = list()
        Wcs.append(nn.Linear(dimc, dimh))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = list()
        Wczs.append(nn.Linear(dimh, dimh))
        for _ in range(num_hidden_layers - 1):
            Wczs.append(nn.Linear(dimh, dimh // 2))
        Wczs.append(nn.Linear(dimh, 1, bias=True))
        self.Wczs = torch.nn.ModuleList(Wczs)
        for Wcz in self.Wczs:
            Wcz.weight.data.zero_()
            Wcz.bias.data.zero_()

        Wcxs = list()
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(nn.Linear(dimh, dimh // 2))
        Wcxs.append(nn.Linear(dimh, 1, bias=True))
        self.Wcxs = torch.nn.ModuleList(Wcxs)
        for Wcx in self.Wcxs:
            Wcx.weight.data.zero_()
            Wcx.bias.data.zero_()

        Wcx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wcx2s.append(nn.Linear(dimh, dimh // 2))
        self.Wcx2s = torch.nn.ModuleList(Wcx2s)
        for Wcx2 in self.Wcx2s:
            Wcx2.weight.data.zero_()
            Wcx2.bias.data.zero_()

        Wccs = list()
        for _ in range(num_hidden_layers - 1):
            Wccs.append(nn.Linear(dimh, dimh))
        self.Wccs = torch.nn.ModuleList(Wccs)

        self.actnorm0 = ActNormNoLogdet(dimh)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh // 2))
        actnorms.append(ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

        actnorm2s = list()
        for _ in range(num_hidden_layers - 1):
            actnorm2s.append(ActNormNoLogdet(dimh // 2))
        self.actnorm2s = torch.nn.ModuleList(actnorm2s)

        self.actnormc = ActNormNoLogdet(dimh)

    def forward(self, x, c):
        c = self.act_c(self.actnormc(self.Wcs[0](c)))
        cz = self.Wczs[0](c) + 1.0
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x) * cz), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x) * cz))

        for Wz, Wx, Wx2, Wcz, Wcx, Wcx2, Wcc, actnorm, actnorm2 in zip(
                self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s,
                self.Wczs[1:-1], self.Wcxs[:-1], self.Wcx2s, self.Wccs,
                self.actnorms[:-1], self.actnorm2s):
            cz = softplus(Wcz(c) + np.exp(np.log(1.0) - 1))
            cx = Wcx(c) + 1.0
            cx2 = Wcx2(c) + 1.0
            c1, c2 = torch.chunk(Wcc(c), 2, dim=1)
            z = self.act(actnorm(Wz(z) * cz + Wx(x) * cx + c1))
            aug = Wx2(x) * cx2 + c2
            aug = actnorm2(aug)
            aug = symm_softplus(aug, self.act) if self.symm_act_first else self.act(aug)
            z = torch.cat([z, aug], 1)

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.actnorms[-1](
            self.Wzs[-1](z) * cz + self.Wxs[-1](x) * cx
        )


# noinspection PyPep8Naming,PyTypeChecker
class DensePICNN(PICNNAbstractClass):
    def __init__(self, dim=2, dimh=16, dimc=2, num_hidden_layers=2, PosLin=PosLinear, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False):
        super(DensePICNN, self).__init__()
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first

        # data path
        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for j in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh * (j + 1), dimh, bias=True))
        Wzs.append(PosLin(dimh * num_hidden_layers, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        # context path
        Wcs = list()
        Wcs.append(nn.Linear(dimc, dimh))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = list()
        for j in range(num_hidden_layers - 1):
            Wczs.append(nn.Linear(dimh, dimh * (j + 1)))
        Wczs.append(nn.Linear(dimh, dimh * num_hidden_layers, bias=True))
        self.Wczs = torch.nn.ModuleList(Wczs)
        for Wcz in self.Wczs:
            Wcz.weight.data.zero_()
            Wcz.bias.data.zero_()

        Wcxs = list()
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(nn.Linear(dimh, dim))
        Wcxs.append(nn.Linear(dimh, dim, bias=True))
        self.Wcxs = torch.nn.ModuleList(Wcxs)
        for Wcx in self.Wcxs:
            Wcx.weight.data.zero_()
            Wcx.bias.data.zero_()

        Wccs = list()
        for _ in range(num_hidden_layers - 1):
            Wccs.append(nn.Linear(dimh, dimh))
        self.Wccs = torch.nn.ModuleList(Wccs)

        self.actnorm0 = ActNormNoLogdet(dimh)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh))
        actnorms.append(ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

        self.actnormc = ActNormNoLogdet(dimh)

    def forward(self, x, c):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        c = self.act_c(self.actnormc(self.Wcs[0](c)))
        for Wz, Wx, Wcz, Wcx, Wcc, actnorm in zip(
                self.Wzs[1:-1], self.Wxs[:-1],
                self.Wczs[:-1], self.Wcxs[:-1], self.Wccs,
                self.actnorms[:-1]):
            cz = softplus(Wcz(c) + np.exp(np.log(1.0) - 1))
            cx = Wcx(c) + 1.0
            z_ = self.act(actnorm(Wz(z * cz) + Wx(x * cx) + Wcc(c)))
            z = torch.cat([z, z_], 1)

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.actnorms[-1](
            self.Wzs[-1](z * cz) + self.Wxs[-1](x * cx)
        )


def test_convicnn():
    import matplotlib.pyplot as plt
    print('Testing convexity')
    icnn = DenseNetICNN(num_hidden_layers=2, num_pooling=2)
    x = torch.randn(64, 1, 28, 28)
    y = torch.randn(64, 1, 28, 28)
    print(np.all((((icnn(x) + icnn(y)) / 2 - icnn((x + y) / 2)) > 0).cpu().data.numpy()))

    print('Visualizing convexity')
    icnn = DenseNetICNN(num_hidden_layers=3, num_pooling=0)
    x = torch.linspace(-10, 10, 100).view(100, 1, 1, 1)
    plt.plot(x.squeeze().numpy(), icnn(x).squeeze().data.numpy())


def test_picnn():
    import matplotlib.pyplot as plt
    print('Testing convexity')
    n = 64
    dim = 123
    dimh = 16
    dimc = 11
    num_hidden_layers = 2
    picnn = PICNN(dim=dim, dimh=dimh, dimc=dimc, num_hidden_layers=num_hidden_layers)
    x1 = torch.randn(n, dim)
    x2 = torch.randn(n, dim)
    c = torch.randn(n, dimc)
    print(np.all((((picnn(x1, c) + picnn(x2, c)) / 2 - picnn((x1 + x2) / 2, c)) > 0).cpu().data.numpy()))

    print('Visualizing convexity')
    dim = 1
    dimh = 16
    dimc = 1
    num_hidden_layers = 2
    picnn = PICNN(dim=dim, dimh=dimh, dimc=dimc, num_hidden_layers=num_hidden_layers)

    c = torch.zeros(1, dimc)
    x = torch.linspace(-10, 10, 100).view(100, 1)
    for c_ in np.linspace(-5, 5, 10):
        plt.plot(x.squeeze().numpy(), picnn(x, c + c_).squeeze().data.numpy())


def plot_softplus():
    import matplotlib.pyplot as plt
    xx = torch.linspace(-4, 4)
    plt.plot(xx.data.numpy(), softplus(xx).data.numpy(), label='Softplus')
    plt.plot(xx.data.numpy(), laplace_softplus(xx).data.numpy(), label='Laplace Softplus')
    plt.plot(xx.data.numpy(), gaussian_softplus(xx).data.numpy(), label='Gaussian Softplus')
    plt.plot(xx.data.numpy(), torch.relu(xx).data.numpy(), label='ReLU')
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig('softplus_functions.png')


if __name__ == '__main__':
    test_picnn()
