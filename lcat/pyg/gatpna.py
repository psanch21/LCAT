from typing import Union, List, Tuple, Optional, Dict
from torch import Tensor

import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.nn.dense.linear import Linear

# from torch_geometric.nn.aggr import DegreeScalerAggregation
from .gatv1 import GATv1Layer
from .gatv2 import GATv2Layer
from torch_geometric.nn import PNAConv


class GATPNAv1Layer(GATv1Layer):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggregators: List[str],
            scalers: List[str],
            deg: Tensor,
            negative_slope: float = 0.2,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'gcn',
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            **kwargs,
    ):
        assert bias == False
        assert 'add_self_loops' not in kwargs
        assert in_channels % heads == 0

        super().__init__(in_channels, in_channels // heads, negative_slope, False,
                         heads, False, mode, share_weights_score,
                         share_weights_value, aggr='add', **kwargs)

        self.aggregators = aggregators
        self.scalers = scalers

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        in_channels = (len(aggregators) * len(scalers) + 1) * in_channels // heads
        self.post_nns = Linear(in_channels * heads, out_channels)
        self.lin = Linear(out_channels, out_channels, bias=bias)
        self.real_out_channels = out_channels

    def update_fn(self, x_agg, x_i):
        x_agg = torch.cat([x_i, x_agg], dim=-1)
        x_agg = self.merge_heads(x_agg)
        x_agg = self.post_nns(x_agg)
        x_agg = self.lin(x_agg)
        return x_agg

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None, convolve: bool = False) -> Tensor:
        if convolve:
            return super().aggregate(inputs, index, dim_size=dim_size)

        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                out_scaler = out
            elif scaler == 'amplification':
                out_scaler = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out_scaler = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out_scaler = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out_scaler = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out_scaler)

        outs = torch.cat(outs, dim=-1)
        return outs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.real_out_channels})')


class GATPNAv2Layer(GATv2Layer):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggregators: List[str],
            scalers: List[str],
            deg: Tensor,
            negative_slope: float = 0.2,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'gcn',
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            **kwargs,
    ):
        assert bias == False
        assert 'add_self_loops' not in kwargs
        assert in_channels % heads == 0
        super().__init__(in_channels, in_channels // heads, negative_slope, False,
                         heads, False, mode, share_weights_score,
                         share_weights_value, aggr='add', **kwargs)

        self.aggregators = aggregators
        self.scalers = scalers

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        in_channels = (len(aggregators) * len(scalers) + 1) * in_channels // heads
        self.post_nns = Linear(in_channels * heads, out_channels)
        self.lin = Linear(out_channels, out_channels, bias=bias)
        self.real_out_channels = out_channels

    def update_fn(self, x_agg, x_i):
        x_agg = torch.cat([x_i, x_agg], dim=-1)
        x_agg = self.merge_heads(x_agg)
        x_agg = self.post_nns(x_agg)
        x_agg = self.lin(x_agg)
        return x_agg

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None, convolve: bool = False) -> Tensor:
        if convolve:
            return super().aggregate(inputs, index, dim_size=dim_size)

        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                out_scaler = out
            elif scaler == 'amplification':
                out_scaler = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out_scaler = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out_scaler = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out_scaler = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out_scaler)

        outs = torch.cat(outs, dim=-1)
        return outs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.real_out_channels})')
