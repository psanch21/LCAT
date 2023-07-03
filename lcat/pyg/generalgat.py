import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import constant
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import set_diag, SparseTensor


# TODO rename arguments
class GeneralGATLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            negative_slope: float = 0.2,
            add_self_loops: bool = True,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'gcn',
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        assert mode in ['gcn', 'gat', 'cat', 'lcat', 'gcngat', 'gatcat']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        self.convolve = mode in ['gcn', 'cat', 'lcat', 'gatcat']

        self.learn_l1 = mode in ['lcat', 'gcngat']
        self.learn_l2 = mode in ['lcat', 'gatcat']

        self.mode = mode
        self.lmbda_ = None
        self.lmbda2_ = None

        if self.learn_l1:
            self.lmbda_ = nn.Parameter(torch.ones([]) * 0, requires_grad=True)
        if self.learn_l2:
            self.lmbda2_ = nn.Parameter(torch.ones([]) * 0, requires_grad=True)

        self.bias = 0.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels * self.heads))

    @property
    def lmbda(self):  # The one that controls GCN-->GATConv
        if self.learn_l1:
            return torch.sigmoid(10 ** self.lmbda_ - 6)
        else:
            if self.mode in ['gcn']:
                return 0.
            elif self.mode in ['gat', 'cat', 'gatcat']:
                return 1.0
            else:
                raise NotImplementedError

    @property
    def lmbda2(self):  # The one that controls GATConv-->GAT
        if self.learn_l2:
            return torch.sigmoid(10 ** (2.2 - self.lmbda2_) - 6)
        else:
            if self.mode in ['gcn', 'gat', 'gcngat']:
                return 0.0
            elif self.mode in ['cat']:
                return 1.
            else:
                raise NotImplementedError

    def reset_parameters(self):
        constant(self.lmbda_, math.log10(6))
        constant(self.lmbda2_, 2.2 - math.log10(6))
        constant(self.bias, 0.)

    def get_x_r(self, x):
        raise NotImplementedError

    def get_x_l(self, x):
        raise NotImplementedError

    def get_x_v(self, x):
        raise NotImplementedError

    def get_x_agg(self, x_l, x_r, edge_index, edge_weight):
        if isinstance(edge_index, Tensor):
            edge_index_no_neigh, edge_weight_no_neigh = remove_self_loops(edge_index, edge_weight)
        elif isinstance(edge_index, SparseTensor):
            edge_index_no_neigh = set_diag(edge_index, 0)
            edge_weight_no_neigh = None
        else:
            raise NotImplementedError

        aggr = self.aggr
        self.aggr = 'add'
        x_lr = torch.cat((x_l, x_r), dim=1)
        x_neig_sum = self.propagate(edge_index_no_neigh, x=(x_lr, x_lr), size=None, convolve=True,
                                    edge_weight=edge_weight_no_neigh)
        self.aggr = aggr

        x_agg = self.lmbda * (x_lr + self.lmbda2 * x_neig_sum)

        # Divide by number of neighbors
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        if isinstance(edge_index, Tensor):
            counts = x_agg.new_zeros((x_agg.size(0),))
            a, b = edge_index_no_neigh[i].unique(return_counts=True)
            counts = counts.scatter_add(0, a, b.float())
        elif isinstance(edge_index, SparseTensor):
            counts = edge_index_no_neigh.sum(dim=j)

        x_agg = x_agg / (1 + self.lmbda2 * counts.unsqueeze(-1).unsqueeze(-1))

        return x_agg

    def merge_heads(self, x):
        return x.flatten(start_dim=-2)

    def compute_score(self, x_i, x_j, index, ptr, size_i):
        raise NotImplementedError

    def fix_parameters(self, partial=False):
        raise NotImplementedError

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, size_target: int = None,
                edge_weight: OptTensor = None, return_attention_info: bool = False):

        assert isinstance(x, Tensor) and x.dim() == 2

        # We apply the linear layer before convolving to avoid numerical errors
        x_l = self.get_x_l(x).view(-1, self.heads, self.out_channels)
        x_r = self.get_x_r(x).view(-1, self.heads, self.out_channels)
        x_v = self.get_x_v(x).view(-1, self.heads, self.out_channels)

        num_nodes = x.size(0)
        if size_target is not None:
            num_nodes = min(num_nodes, size_target)

        if self.convolve:
            x_agg = self.get_x_agg(x_l=x_l, x_r=x_r, edge_index=edge_index, edge_weight=edge_weight)
            x_l, x_r = x_agg[:, :self.heads], x_agg[:, self.heads:]

        x_r = x_r[:num_nodes]

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index, 1.)

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        x_lr = [x_l, x_r]
        x_lr[j] = torch.cat((x_lr[j], x_v), dim=-1)

        out = self.propagate(edge_index, x=x_lr, edge_weight=edge_weight, size=None)

        alpha = self._alpha
        del self._alpha

        score = self._score
        del self._score

        out = self.update_fn(x_agg=out, x_i=x_v)

        if return_attention_info:
            assert alpha is not None
            return out, (edge_index, alpha), score
        else:
            return out

    def update_fn(self, x_agg, x_i):
        return self.merge_heads(x_agg) + self.bias

    def message(self, x_j: Tensor,
                x_i: Tensor, index: Tensor,
                ptr: OptTensor,
                size_i: Optional[int],
                edge_weight: OptTensor,
                convolve=False) -> Tensor:
        if convolve:
            return x_j

        s = x_i.size(-1)
        x_j, x_v = x_j[..., :s], x_j[..., s:]

        score = self.compute_score(x_i, x_j, index, ptr, size_i)
        self._alpha = softmax(score, index, ptr, size_i)
        self._score = score

        num_neighbors = softmax(torch.ones_like(score), index, ptr, size_i).reciprocal()

        edge_weight = 1. if edge_weight is None else edge_weight.view(-1, 1, 1)
        return x_v * (self._alpha * num_neighbors) * edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class GeneralGATConv(nn.Module):
    def __init__(self, LayerName, dim_in, dim_out, bias=False, add_attn_info=False, **kwargs):
        super(GeneralGATConv, self).__init__()

        self.model = LayerName(dim_in, dim_out, bias=bias, **kwargs)
        self.add_attn_info = add_attn_info

    def forward(self, batch):
        if self.add_attn_info:
            out, (edge_index, alpha), score = self.model(batch.node_feature,
                                                         batch.edge_index,
                                                         return_attention_info=True)
            batch.node_feature = out
            if hasattr(batch, 'edges_split'):
                alpha_split = alpha[batch.edges_split]
            else:
                alpha_split = alpha
            cond_1 = (batch.edge_label == 1).flatten()
            cond_0 = (batch.edge_label == 0).flatten()

            setattr(batch, 'alpha_mean', torch.mean(alpha_split))
            setattr(batch, 'alpha_std', torch.std(alpha_split))
            setattr(batch, 'alpha_mean_1', torch.mean(alpha_split[cond_1, :]))
            setattr(batch, 'alpha_std_1', torch.std(alpha_split[cond_1, :]))
            setattr(batch, 'alpha_mean_0', torch.mean(alpha_split[cond_0, :]))
            setattr(batch, 'alpha_std_0', torch.std(alpha_split[cond_0, :]))
            setattr(batch, 'score', score)

            if hasattr(batch, 'edge_label_ind'):
                classes = batch.edge_label_ind.unique()

                for c_src in classes:
                    for c_dst in classes:
                        tmp = torch.tensor([c_src, c_dst])
                        cond = (batch.edge_label_ind == tmp).sum(1) == 2
                        my_str = f'alpha_mean_{int(c_src.item())}{int(c_dst.item())}'
                        setattr(batch, my_str, torch.mean(alpha_split[cond, ...]))
        else:
            batch.node_feature = self.model(batch.node_feature, batch.edge_index)

        batch.lmbda2 = float(self.model.lmbda2)
        batch.lmbda = float(self.model.lmbda)

        if hasattr(self.model, 'eps'):
            batch.eps = self.model.eps

        return batch
