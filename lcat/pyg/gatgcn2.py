from math import log
from typing import Union, Tuple

import torch
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor

from .gatv1 import GATv1Layer
from .gatv2 import GATv2Layer


class GAT2v1Layer(GATv1Layer):
    x_0: OptTensor

    def __init__(
            self,
            channels: Union[int, Tuple[int, int]],
            alpha: float,
            theta: float = None,
            layer: int = None,
            normalize: bool = True,
            add_self_loops: bool = True,
            negative_slope: float = 0.2,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'gcn',
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            **kwargs,
    ):
        assert share_weights_value  # TODO

        kwargs.setdefault('aggr', 'add')
        super().__init__(channels, channels // heads, negative_slope, add_self_loops,
                         heads, bias, mode, share_weights_score,
                         share_weights_value, **kwargs)
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.normalize = normalize
        self.x_0 = None

    def get_x_v(self, x):
        return x

    def forward(self, x: Union[Tensor, PairTensor], x_0: Tensor, edge_index: Adj, size_target: int = None,
                edge_weight: OptTensor = None, return_attention_info: bool = False):
        assert not return_attention_info  # TODO

        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

        self.x_0 = x_0
        return super().forward(x, edge_index, size_target, edge_weight, return_attention_info)

    def update_fn(self, x_agg, x_i):
        x_0 = self.x_0
        x_agg = self.merge_heads(x_agg)

        x_agg.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x_agg.size(0)]

        x_agg = x_agg.add_(x_0)
        x_agg = torch.addmm(x_agg, x_agg, self.lin_v.weight, beta=1. - self.beta, alpha=self.beta)

        return x_agg


class GAT2v2Layer(GATv2Layer):
    x_0: OptTensor

    def __init__(
            self,
            channels: Union[int, Tuple[int, int]],
            alpha: float,
            theta: float = None,
            layer: int = None,
            normalize: bool = True,
            add_self_loops: bool = True,
            negative_slope: float = 0.2,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'gcn',
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            **kwargs,
    ):
        assert share_weights_value  # TODO

        kwargs.setdefault('aggr', 'add')
        super().__init__(channels, channels // heads, negative_slope, add_self_loops,
                         heads, bias, mode, share_weights_score,
                         share_weights_value, **kwargs)
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.normalize = normalize
        self.x_0 = None

    def get_x_v(self, x):
        return x

    def forward(self, x: Union[Tensor, PairTensor], x_0: Tensor, edge_index: Adj, size_target: int = None,
                edge_weight: OptTensor = None, return_attention_info: bool = False):
        assert not return_attention_info  # TODO

        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops,  dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

        self.x_0 = x_0
        return super().forward(x, edge_index, size_target, edge_weight, return_attention_info)

    def update_fn(self, x_agg, x_i):
        x_0 = self.x_0
        x_agg = self.merge_heads(x_agg)

        x_agg.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x_agg.size(0)]

        x_agg = x_agg.add_(x_0)
        x_agg = torch.addmm(x_agg, x_agg, self.lin_v.weight, beta=1. - self.beta, alpha=self.beta)

        return x_agg
