import os

import pytest
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from lcat.pyg import GATv1Layer, GATv2Layer
from lcat.pyg import GAT2v1Layer, GAT2v2Layer
from lcat.pyg import GATPNAv1Layer, GATPNAv2Layer


def assert_lcat(output, batch, out_channels, heads):
    assert output.ndim == 2
    assert output.shape[0] == batch.x.shape[0]
    assert output.shape[-1] == out_channels * heads


@pytest.fixture(scope="session")
def dataset():
    root = os.path.join('..', 'Data')
    dataset = TUDataset(root=root,
                        name='MUTAG',
                        transform=None,
                        pre_transform=None,
                        pre_filter=None,
                        use_node_attr=True,
                        use_edge_attr=True,
                        cleaned=True
                        )
    return dataset


@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("heads_aggr", ['add', 'mean', 'concat'])
@pytest.mark.parametrize("mode", ['gcn', 'gat', 'cat', 'lcat', 'gcngat', 'gatcat'])
@pytest.mark.parametrize("att_version", ['v1', 'v2'])
def test_lcat_module_GAT_GCN(out_channels,
                             add_self_loops,
                             heads,
                             heads_aggr,
                             mode,
                             att_version,
                             dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    in_channels = batch.x.shape[1]

    if att_version == 'v1':
        LCAT = GATv1Layer
    elif att_version == 'v2':
        LCAT = GATv2Layer
    lcat = LCAT(
        in_channels=in_channels,
        out_channels=out_channels,
        negative_slope=0.2,
        add_self_loops=add_self_loops,
        heads=heads,
        bias=True,
        mode=mode,
        share_weights_score=False,
        shere_weights_value=False,
        aggr='mean',
    )
    output = lcat(x=batch.x,
                  edge_index=batch.edge_index)

    assert_lcat(output, batch, out_channels, heads)



@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("heads_aggr", ['add', 'mean', 'concat'])
@pytest.mark.parametrize("mode", ['gcn', 'gat', 'cat', 'lcat', 'gcngat', 'gatcat'])
@pytest.mark.parametrize("att_version", ['v1', 'v2'])
def test_lcat_module_GAT_GCN2(out_channels,
                              add_self_loops,
                              heads,
                              heads_aggr,
                              mode,
                              att_version,
                              dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    in_channels = batch.x.shape[1]

    if att_version == 'v1':
        LCAT = GAT2v1Layer
    elif att_version == 'v2':
        LCAT = GAT2v2Layer
    lcat = LCAT(
        channels=out_channels,
        alpha=0.4,
        theta=0.1,
        layer=1,
        normalize=True,
        add_self_loops=add_self_loops,
        heads=heads,
        bias=True,
        mode=mode,
        share_weights_score=True,
        share_weights_value=True,
        aggr='mean',
    )

    mlp = nn.Linear(in_channels, out_channels)

    batch.x = mlp(batch.x)
    output = lcat(x=batch.x,
                  x_0=batch.x,
                  edge_index=batch.edge_index)

    assert_lcat(output, batch, out_channels, 1)




from torch_geometric.nn import PNAConv


@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("heads_aggr", ['add', 'mean', 'concat'])
@pytest.mark.parametrize("mode", ['gcn', 'gat', 'cat', 'lcat', 'gcngat', 'gatcat'])
@pytest.mark.parametrize("att_version", ['v1', 'v2'])
def test_lcat_module_GAT_PNA(out_channels,
                             add_self_loops,
                             heads,
                             heads_aggr,
                             mode,
                             att_version,
                             dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    in_channels = batch.x.shape[1]

    if att_version == 'v1':
        LCAT = GATPNAv1Layer
    elif att_version == 'v2':
        LCAT = GATPNAv2Layer
    else:
        raise NotImplementedError
    lcat = LCAT(
        in_channels=out_channels,
        out_channels=out_channels,
        aggregators=['mean', 'min', 'max'],
        scalers=['identity', 'amplification', 'attenuation'],
        deg=PNAConv.get_degree_histogram(loader),
        negative_slope=0.2,
        heads=heads,
        bias=False,
        mode=mode,
        share_weights_score=False,
        shere_weights_value=False,
    )
    mlp = nn.Linear(in_channels, out_channels)

    batch.x = mlp(batch.x)
    output = lcat(x=batch.x,
                  edge_index=batch.edge_index)

    assert_lcat(output, batch, out_channels, 1)


