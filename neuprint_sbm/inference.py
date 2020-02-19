import os
import pickle
import logging
import argparse

import numpy as np
import pandas as pd

import graph_tool.inference
import graph_tool as gt

from dvidutils import LabelMapper

from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC

from neuclease.util import Timer

from neuclease import configure_default_logging
configure_default_logging()


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuprint-server', '-n', default='neuprint.janelia.org')
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--init', '-i', choices=['groundtruth', 'random'])
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--min-weight', '-w', default=10, type=int)
    args = parser.parse_args()

    c = Client(args.neuprint_server, args.dataset)
    export_dir = f"{c.dataset}-w{args.min_weight}-from-{args.init}"
    os.makedirs(export_dir, exist_ok=True)

    # Fetch connectome (and export)
    with Timer("Fetching/exporting connectome", logger):
        criteria = NC(status='Traced', cropped=False, client=c)
        neuron_df, roi_conn_df = fetch_adjacencies(criteria, criteria, min_total_weight=args.min_weight, export_dir=export_dir, properties=['type', 'instance'], client=c)
        conn_df = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
    
    strong_connections_df, g, nbs, partition_df = infer_hierarchy(neuron_df,
                                                                  conn_df,
                                                                  args.min_weight,
                                                                  args.init,
                                                                  args.verbose,
                                                                  args.debug)

    with Timer("Exporting inference results", logger):
        pickle.dump(g,                     open(f'{export_dir}/graph.pkl', 'wb'))
        pickle.dump(nbs,                   open(f'{export_dir}/nested-block-state.pkl', 'wb'))
        pickle.dump(partition_df,          open(f'{export_dir}/partition_df.pkl', 'wb'))
        pickle.dump(strong_connections_df, open(f'{export_dir}/strong_connections_df.pkl', 'wb'))

    logger.info("DONE")

def load_table(df):
    if isinstance(df, str):
        ext = df[-4:]
        assert ext in ('.csv', '.npy', '.pkl')
        if ext == '.csv':
            df = pd.read_csv(df)
        elif ext =='.npy':
            df = pd.DataFrame(np.load(df, allow_pickle=True))
        elif ext in '.pkl':
            df = pickle.load(open(df, 'rb'))

    assert isinstance(df, pd.DataFrame)
    return df
    

def infer_hierarchy(neuron_df, connection_df, min_weight=10, init='groundtruth', verbose=True, special_debug=False):
    ##
    ## TODO: If filtering connections for min_weight drops some neurons entirely, they should be removed from neuron_df
    ##
    lsf_slots = os.environ.get('LSB_DJOB_NUMPROC', default=0)
    if lsf_slots:
        os.environ['OMP_NUM_THREADS'] = lsf_slots
        logger.info(f"Using {lsf_slots} CPUs for OpenMP")

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
        with Timer("Computing initial hierarchy from groundtruth", logger):
            assign_morpho_indexes(neuron_df)
            num_morpho_groups = neuron_df.morpho_index.max()+1
            init_bs = [neuron_df['morpho_index'].values, np.zeros(num_morpho_groups, dtype=int)]
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
        nbs = graph_tool.inference.minimize_nested_blockmodel_dl(g,
                                                                 bs=init_bs,
                                                                 mcmc_args=dict(parallel=True), # see graph-tool docs and mailing list for caveats 
                                                                 deg_corr=True,
                                                                 verbose=verbose)

    partition_df = construct_partition_table(nbs, neuron_df, vertexes, vertex_reverse_mapper)
    return strong_connections_df, g, nbs, partition_df


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
    
    ndf = neuron_df.rename(columns={'bodyId': 'body'})
    partition_df['body'] = vertex_reverse_mapper.apply(partition_df.loc[:,0].values)
    partition_df = partition_df.merge(ndf[['body', 'instance']], 'left', on='body')
    partition_df = partition_df.merge(ndf[['body', 'type']], 'left', on='body')
    
    partition_df['instance'] = partition_df['instance'].fillna('')
    partition_df['type'] = partition_df['type'].fillna('')
    
    # Sort by hierarchy top-to-bottom (except level 0), then by instance and body
    partition_df = partition_df.sort_values([*range(num_levels-1, 0, -1), 'instance', 'body']).reset_index(drop=True)
    
    # Put 'instance' and 'body' first
    partition_df = partition_df[['instance', 'type', 'body', *range(num_levels)]]

    return partition_df


def assign_morpho_indexes(neurons_df):
    neurons_df['morpho_type'] = neurons_df['type'].fillna('').apply(lambda s: s.split('_')[0])
    morpho_types = neurons_df['morpho_type'].sort_values().reset_index(drop=True)
    morpho_types = morpho_types.loc[morpho_types != '']

    morpho_mapping = dict(zip(morpho_types, morpho_types.index))
    morpho_mapping[''] = len(morpho_types)
    neurons_df['morpho_index'] = neurons_df['morpho_type'].map(morpho_mapping)
    
    # Everything with no morpho type at all gets a group by itself.
    empty = (neurons_df['morpho_type'] == '')
    neurons_df.loc[empty, 'morpho_index'] = len(morpho_types) + np.arange(empty.sum())


if __name__ == "__main__":
    main()
