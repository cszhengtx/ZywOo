import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 30., w0_initial = 30., use_bias = True, final_activation = None, weight_decay = 2, dropout = False, ratio = 0.75):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.weight_decay = weight_decay

        self.layers = nn.ModuleList([])
        self.decay_layers = nn.ModuleList([])
        for ind in range(num_layers-weight_decay):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))
            if dropout:
                self.layers.append(nn.Dropout(ratio))

        for ind in range(weight_decay):
            self.decay_layers.append(Siren(dim_in=dim_hidden, dim_out=dim_hidden, w0=layer_w0, use_bias=use_bias, is_first=is_first))


        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers-self.weight_decay)    #use for latent if input the noise img. now it's false
        out_arr = []
        for i, (layer, mod) in enumerate(zip(self.layers, mods)):
            x = layer(x)
            out_arr.append(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')


        mods_ = cast_tuple(mods, self.weight_decay)  # use for latent if input the noise img. now it's false
        for i, (layer, mod) in enumerate(zip(self.decay_layers, mods_)):
            #x = (x + out_arr[i + 1]) / 2
            x = layer(x)
            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x), out_arr

# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)

# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss()

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps=image_height), torch.linspace(-1, 1, steps=image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

        # tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        # mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        # mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        # self.register_buffer('grid', mgrid)

    def forward(self, coordinates, features = None, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = coordinates #self.grid.clone().detach().requires_grad_()
        out, feat_arr = self.net(coords, mods)

        #out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(features):
            return self.MSELoss(features, out), out, feat_arr

        return out



