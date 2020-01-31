import os
import re
import zlib
import pickle
import logging
import subprocess

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from matplotlib.colors import hsv_to_rgb

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool, LabelSet, TapTool
from bokeh.events import Tap
#from bokeh.layouts import row, column
import bokeh.layouts as bl
import hvplot.pandas

import graph_tool.inference
import graph_tool as gt

from dvidutils import LabelMapper

from neuprint import Client

from neuclease.util import Timer, tqdm_proxy

from neuclease import configure_default_logging
configure_default_logging()



#from neuclease.dvid import *
#output_notebook()

logger = logging.getLogger(__name__)


def load_table(df):
    if isinstance(df, str):
        ext = df[-4:]
        assert ext in ('.csv', '.npy')
        if ext == '.csv':
            df = pd.read_csv(df)
        elif ext =='.npy':
            df = pd.DataFrame(np.load(df, allow_pickle=True))

    assert isinstance(df, pd.DataFrame)
    return df
    

def infer_hierarchy(neuron_df, connection_df, min_weight=10, init='groundtruth', verbose=True, special_debug=False):
    assert init in ('groundtruth', 'random')
    neuron_df = load_table(neuron_df)
    connection_df = load_table(connection_df)

    assert {*neuron_df.columns} >= {'bodyId', 'instance', 'type'}
    assert {*connection_df.columns} >= {'bodyId_pre', 'bodyId_post', 'weight'}

    if special_debug:
        # Choose a very small subset of the data
        neuron_df = neuron_df.iloc[::100]
        bodies = neuron_df['bodyId']
        connection_df = connection_df.query('bodyId_pre in @bodies and bodyId_post in @bodies')

    if init == "groundtruth":
        assign_type_levels(neuron_df)
        init_bs = type_blocks(neuron_df)
    else:
        init_bs = None

    # If this is a per-ROI table, sum up the ROIs.
    if 'roi' in connection_df:
        connection_df = connection_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

    strong_connections_df = connection_df.query('weight >= @min_weight')
    strong_bodies = pd.unique(strong_connections_df[['bodyId_pre', 'bodyId_post']].values.reshape(-1))
    weights = strong_connections_df.set_index(['bodyId_pre', 'bodyId_post'])['weight']
    
    logger.info(f"Strong connectome (cutoff={min_weight}) has {len(strong_bodies)} bodies and {len(weights)} edges")
    
    vertexes = np.arange(len(strong_bodies), dtype=np.uint32)
    vertex_mapper = LabelMapper(strong_bodies.astype(np.uint64), vertexes)
    vertex_reverse_mapper = LabelMapper(vertexes, strong_bodies.astype(np.uint64))

    g = construct_graph(weights, vertexes, vertex_mapper)
    
    with Timer("Running inference"):
        # Computes a NestedBlockState
        nbs = graph_tool.inference.minimize_nested_blockmodel_dl(g, bs=init_bs, deg_corr=True, verbose=verbose)

    partition_df = construct_partition_table(nbs, neuron_df, vertexes, vertex_reverse_mapper)
    return g, nbs, partition_df


def construct_graph(weights, vertexes, vertex_mapper):
    #assert weights.index.shape[1] == 2, \
    #    "Please pass a series, indexed by [bodyId_pre, bodyId_post]"
    
    # Construct graph from strong connectome
    multi_edges = []
    for (bodyId_pre, bodyId_post), weight in weights.iteritems():
        multi_edges.extend([(bodyId_pre, bodyId_post)]*weight)
    
    multi_edges = np.asarray(multi_edges, dtype=np.uint64)
    edges = vertex_mapper.apply(multi_edges)
    assert edges.dtype == np.uint32
    
    g = gt.Graph(directed=True)
    g.add_vertex(np.uint32(len(vertexes)-1))
    g.add_edge_list(edges)
    return g


def construct_partition_table(nbs, neuron_df, vertexes, vertex_reverse_mapper):
    """
    Args:
        nbs (NestedBlockState):
            The result of the SBM inference
        vertexes:
            The node IDs used in the inference
        vertex_reverse_mapper (LabelMapper):
            Maps from vertexes back to bodyIds
    
    Returns (DataFrame):
        The partition table, which has columns:
        ['instance', 'type', 'body', 0, 1, 2, ..., N]
        Where the columns named with integers (named with actual ints, not strings)
        correspond to which block each body belongs to at every level of the hierarchy.
        At level 0, the 'block' is simply the vertex itself.
        Level N corresponds to the top of the hierarchy, so all bodies belong to the same block at that level (block 0).
     
    """
    num_levels = 1+len(nbs.get_bs())
    node_partitions = np.zeros((len(vertexes), num_levels), dtype=np.uint32)
    node_partitions[:, 0] = vertexes
    for level, bs in enumerate(nbs.get_bs(), start=1):
        if level == 1:
            node_partitions[:, 1] = bs[:]
            continue
        for lower_partition, partition in enumerate(bs):
            node_partitions[(node_partitions[:, level-1] == lower_partition), level] = partition
    
    partition_df = pd.DataFrame(node_partitions, columns=range(num_levels))
    
    partition_df['body'] = vertex_reverse_mapper.apply(partition_df.loc[:,0].values)
    partition_df = partition_df.merge(neuron_df['instance'], 'left', left_on='body', right_index=True)
    partition_df = partition_df.merge(neuron_df['type'], 'left', left_on='body', right_index=True)
    
    partition_df['instance'] = partition_df['instance'].fillna('')
    partition_df['type'] = partition_df['type'].fillna('')
    
    # Sort by hierarchy top-to-bottom (except level 0), then by instance and body
    partition_df = partition_df.sort_values([*range(num_levels-1, 0, -1), 'instance', 'body']).reset_index(drop=True)
    
    # Put 'instance' and 'body' first
    partition_df = partition_df[['instance', 'type', 'body', *range(num_levels)]]

    return partition_df


def assign_type_levels(neurons_df):
    # Example name: ADL01oa_pct
    nums = '[0-9]*'
    chars = '[^0-9_]*'
    pat = re.compile(f'_?({chars})?({nums})?(_?{chars})?(_?{chars}{nums})?(_?{chars}{nums}_?)?')

    def level_name(ntype, level):
        assert level >= 1
        if not isinstance(ntype, str):
            return ''
        ntype = ntype.replace('(', '_')
        ntype = ntype.replace(')', '_')
        if ntype.endswith('pct'):
            ntype = ntype[:-3]
        if ntype.endswith('_'):
            ntype = ntype[:-1]
        return ''.join(re.match(pat, ntype).groups()[:6-level])

    for level in range(5,0,-1):
        neurons_df[f'type_{level}'] = neurons_df['type'].apply(lambda s: level_name(s, level))
    neurons_df.sort_values([f'type_{i}' for i in range(1,6)], inplace=True)


def type_blocks(neurons_df):
    neurons_df.reset_index(drop=True, inplace=True)
    
    t1_names = pd.unique(neurons_df['type_1'])
    t2_names = pd.unique(neurons_df['type_2'])
    t3_names = pd.unique(neurons_df['type_3'])
    t4_names = pd.unique(neurons_df['type_4'])
    t5_names = pd.unique(neurons_df['type_5'])

    t1_lookup = pd.Series(index=t1_names, data=np.arange(len(t1_names)))
    t2_lookup = pd.Series(index=t2_names, data=np.arange(len(t2_names)))
    t3_lookup = pd.Series(index=t3_names, data=np.arange(len(t3_names)))
    t4_lookup = pd.Series(index=t4_names, data=np.arange(len(t4_names)))
    t5_lookup = pd.Series(index=t5_names, data=np.arange(len(t5_names)))
    
    neurons_df['block_1'] = 0
    neurons_df['block_2'] = 0
    neurons_df['block_3'] = 0
    neurons_df['block_4'] = 0
    neurons_df['block_5'] = 0

    for row in tqdm_proxy(neurons_df.itertuples(), total=len(neurons_df)):
        neurons_df.loc[row.Index, 'block_1'] = t1_lookup[row.type_1]
        neurons_df.loc[row.Index, 'block_2'] = t2_lookup[row.type_2]
        neurons_df.loc[row.Index, 'block_3'] = t3_lookup[row.type_3]
        neurons_df.loc[row.Index, 'block_4'] = t4_lookup[row.type_4]
        neurons_df.loc[row.Index, 'block_5'] = t5_lookup[row.type_5]

    blocks = []
    blocks.append(neurons_df['block_1'].values)
    blocks.append(neurons_df.drop_duplicates(['block_1']).sort_values('block_1')['block_2'].values)
    blocks.append(neurons_df.drop_duplicates(['block_2']).sort_values('block_2')['block_3'].values)
    blocks.append(neurons_df.drop_duplicates(['block_3']).sort_values('block_3')['block_4'].values)
    blocks.append(np.zeros((neurons_df['block_4'].nunique(),), dtype=int))
    return blocks


if __name__ == "__main__":
    g, nbs, partition_df = infer_hierarchy('traced-adjacencies-2020-01-30/traced-neurons.csv',
                                           'traced-adjacencies-2020-01-30/traced-total-connections.csv',
                                           special_debug=False)

    os.makedirs('inferred-blocks', exist_ok=True)
    pickle.dump(g,              open('inferred-blocks/graph.pkl', 'wb'))
    pickle.dump(nbs,            open('inferred-blocks/nested-block-state.pkl', 'wb'))
    pickle.dump(partition_df,   open('inferred-blocks/partition_df.pkl', 'wb'))
