from typing import Union, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear, Parameter
from torch_geometric.nn.inits import constant

from .generalgat import GeneralGATLayer


class GATv2Layer(GeneralGATLayer):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            negative_slope: float = 0.2,
            add_self_loops: bool = True,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'gcn',
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(in_channels,
                         out_channels,
                         negative_slope,
                         add_self_loops,
                         heads,
                         bias,
                         mode,
                         aggr,
                         **kwargs)

        self.lin_l = Linear(in_channels, out_channels * self.heads, bias=bias, weight_initializer='glorot')
        if share_weights_score:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, out_channels * self.heads, bias=bias, weight_initializer='glorot')

        if share_weights_value:
            self.lin_v = self.lin_l if self.flow == 'source_to_target' else self.lin_r
        else:
            self.lin_v = Linear(in_channels, out_channels * self.heads, bias=bias, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att = Parameter(torch.Tensor(1, self.heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        constant(self.att, 1.)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_v.reset_parameters()

    def get_x_r(self, x):
        return self.lin_r(x)

    def get_x_l(self, x):
        return self.lin_l(x)

    def get_x_v(self, x):
        return self.lin_v(x)

    def compute_score(self, x_i, x_j, index, ptr, size_i):
        return torch.sum(F.leaky_relu(x_i + x_j, self.negative_slope) * self.att, dim=-1, keepdim=True)
