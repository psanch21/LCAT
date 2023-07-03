from typing import Union, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from torch_geometric.nn.dense.linear import Linear, Parameter
from torch_geometric.nn.inits import constant


class GATv1Layer(nn.Module):

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            negative_slope: float = 0.2,
            add_self_loops: bool = True,
            heads: int = 1,
            bias: bool = True,
            convolve: bool = True,
            lambda_policy: str = None,  # [None, 'learn1', 'learn2', 'learn12', 'gcn_gat']
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            gcn_mode: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.convolve = convolve
        self.lambda_policy = lambda_policy
        self.gcn_mode = gcn_mode
        assert out_channels % heads == 0, f'{out_channels} {heads}'
        assert not gcn_mode or (gcn_mode and self.convolve)
        assert lambda_policy in [None, 'learn1', 'learn2', 'learn12', 'gcn_gat']

        self.lin_l = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')
        if share_weights_score:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')

        if share_weights_value:
            self.lin_v = self.lin_l
        else:
            self.lin_v = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_l = Parameter(torch.Tensor(1, self.heads, out_channels // self.heads))
        self.att_r = Parameter(torch.Tensor(1, self.heads, out_channels // self.heads))

        self.lmbda_ = None
        self.lmbda2_ = None
        if self.lambda_policy is not None:
            self.lmbda_ = nn.Parameter(torch.ones([]) * 0, requires_grad=True)
            self.lmbda2_ = nn.Parameter(torch.ones([]) * 0, requires_grad=True)

        self.bias = 0.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # Alternative initilization in case it doesn't work
        #gain = nn.init.calculate_gain('relu')
        #nn.init.xavier_normal_(self.lin_l.weight, gain=gain)
        #nn.init.xavier_normal_(self.lin_r.weight, gain=gain)
        #nn.init.xavier_normal_(self.lin_v.weight, gain=gain)
        #nn.init.xavier_normal_(self.att_r, gain=gain)
        #nn.init.xavier_normal_(self.att_l, gain=gain)

        constant(self.att_l, 1.)
        constant(self.att_r, 1.)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_v.reset_parameters()

        constant(self.lmbda_, math.log10(6))
        constant(self.lmbda2_, 1.4)
        constant(self.bias, 0.)

    @property
    def lmbda(self):  # The one that controls GCN-->GATConv
        if self.gcn_mode:
            return 0.
        if self.lambda_policy is None:
            return 1.
        elif self.lambda_policy == 'learn2':
            return 1.

        return torch.sigmoid(10 ** self.lmbda_ - 6)

    @property
    def lmbda2(self):  # The one that controls GATConv-->GAT
        if self.lambda_policy is None:
            return 1.
        elif self.lambda_policy == 'learn1':
            return 1.
        elif self.lambda_policy == 'gcn_gat':
            return 0.

        return torch.sigmoid(10 ** (2.2 - self.lmbda2_) - 6)

    def forward(self, graph, x):
        with graph.local_scope():
            # assert not graph.is_block

            x_l = self.lin_l(x).view(-1, self.heads, self.out_channels // self.heads) * self.att_l
            x_r = self.lin_r(x).view(-1, self.heads, self.out_channels // self.heads) * self.att_r
            x_v = self.lin_v(x).view(-1, self.heads, self.out_channels // self.heads)

            x_l = x_l.sum(dim=-1, keepdim=True)
            x_r = x_r.sum(dim=-1, keepdim=True)

            if self.convolve:
                x_lr = torch.cat((x_l, x_r), dim=1)

                graph.srcdata.update({'hsrc': x_lr})  # (num_nodes, num_heads, num_feats)
                graph.dstdata.update({
                    'hdst': 0. * x_lr[:graph.number_of_dst_nodes()]  # (num_nodes, num_heads, num_feats)
                })

                graph.update_all(fn.u_add_v('hsrc', 'hdst', 'hconv'), fn.sum('hconv', 'hconvsum'))

                num_neigh = graph.in_degrees().unsqueeze(-1).unsqueeze(-1)
                x_agg = self.lmbda * graph.dstdata.pop('hconvsum')

                diff = graph.num_src_nodes() - graph.num_dst_nodes()
                if diff > 0:
                    zeroes = x_agg.new_zeros((diff,) + x_agg.shape[1:])
                    x_agg = torch.cat((x_agg, zeroes), dim=0)
                    zeroes = num_neigh.new_zeros((diff, ) + num_neigh.shape[1:])
                    num_neigh = torch.cat((num_neigh, zeroes), dim=0)

                x_agg = self.lmbda2 * (graph.srcdata['hsrc'] + x_agg)
                x_agg = x_agg / (1 + self.lmbda * num_neigh)

                x_l, x_r = x_agg[:, :self.heads], x_agg[:, self.heads:]

            assert x.dim() == 2

            if graph.is_block:
                x_r = x_r[:graph.number_of_dst_nodes()]

            # (num_src_edge, num_heads, out_dim)
            graph.srcdata.update({'el': x_l})
            graph.srcdata.update({'ev': x_v})
            graph.dstdata.update({'er': x_r})

            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            score = F.leaky_relu(graph.edata.pop('e'), self.negative_slope)  # (num_src_edge, num_heads, out_dim)
            graph.edata['a'] = edge_softmax(graph, score)

            graph.update_all(fn.u_mul_e('ev', 'a', 'm'), fn.sum('m', 'ft'))
            out = graph.dstdata['ft']

            return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
