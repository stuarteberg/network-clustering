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


from .inference import load_table

#from neuclease.dvid import *
#output_notebook()

logger = logging.getLogger(__name__)


def load_pickle(x):
    if isinstance(x, str):
        assert x.endswith('.pkl')
        x = pickle.load(open(x, 'rb'))
    return x


class HierarchyBroswer:
    
    def __init__(self, graph, nbs, partition_df, neuron_df, strong_connections_df):
        with Timer("Loading data", logger):
            graph = load_pickle(graph)
            nbs = load_pickle(nbs)
            partition_df = load_table(partition_df)
            neuron_df = load_table(neuron_df)
            strong_connections_df = load_table(strong_connections_df)
        
        num_levels = len(nbs.get_bs())+1
        assert {*partition_df.columns} > {*range(num_levels)}
        assert num_levels not in partition_df, \
            "partition_df does not match NestedBlockState levels"

        self.graph = graph
        self.nbs = nbs
        self.num_levels = num_levels
        self.partition_df = partition_df
        self.neuron_df = neuron_df
        self.strong_connections_df = strong_connections_df
        
        with Timer("Initialzing browser", logger):
            self._initialize()
   
    def show_notebook(self, notebook_url='http://localhost:8888'):
        output_notebook()
        show(self.modify_doc, notebook_url=notebook_url) 

   
    def _initialize(self):
        # Compute Y-midpoint of each block at every level
        group_midpoints = []
        for level in range(self.num_levels):
            midpoints = {}
            for block, group_df in self.partition_df.groupby(level, sort=False)[[]]:
                midpoints[block] = (group_df.index.min() + group_df.index.max()) / 2
            group_midpoints.append(midpoints)

        # Construct line segment endpoints that will draw the tree hierarchy
        all_group_modes = {}
        
        # At the bottom level, every node is its own group, so the 'mode' is trivial.
        all_group_modes[0] = self.partition_df[[0, 'type']].rename(columns={0: 'node', 'type': 'type_mode'})
        all_group_modes[0]['level'] = 0
        
        tree_line_segments = []
        for level in tqdm_proxy(range(1, self.num_levels)):
            #lower_group_points = []
            for lower_node, node in enumerate(self.nbs.get_bs()[level-1]):
                left_x = level-1
                right_x = level
                left_y = group_midpoints[level-1][lower_node]
                right_y = group_midpoints[level][node]
        
                tree_line_segments.append([level-1, lower_node, node, left_x, left_y, right_x, right_y])
        
            # Drop null types before finding most common,
            # so that we find "most common non-null type"
            modes = (self.partition_df[['type', level]]
                         .query("type != ''")
                         .groupby(level)['type']
                         .apply(lambda s: s.mode().iloc[0])
                         .rename('type_mode'))
        
            # Since type-less nodes were dropped above, nodes which are empty "all the way down" are missing.
            # Add them back in, with an empty string.
            modes = (self.partition_df[[level]].drop_duplicates()
                         .merge(modes, 'left', left_on=level, right_index=True)
                         .fillna('')
                         .set_index(level)['type_mode'])
            
            all_group_modes[level] = modes
            all_group_modes[level].index.name = 'node'
            all_group_modes[level] = all_group_modes[level].reset_index()
            all_group_modes[level]['level'] = level
        
        # Add a 0-length line segment for the root node
        root_x = self.num_levels-1
        root_y = tree_line_segments[-1][-1]
        tree_line_segments.append([self.num_levels-1, 0, -1, root_x, root_y, root_x, root_y ])
        tree_line_segments_df = pd.DataFrame(tree_line_segments, columns=['level', 'node', 'parent', 'x', 'y', 'parent_x', 'parent_y'])
        
        all_stats_df = pd.concat(all_group_modes.values())
        all_stats_df = all_stats_df.merge(tree_line_segments_df, 'left', on=['level', 'node'])
        all_stats_df['color'] = 'gray'
        all_stats_df = all_stats_df[['level', 'node', 'parent', 'x', 'y', 'parent_x', 'parent_y', 'type_mode', 'color']]
    

        # Initialize strengths with 0 width lines (will be updated dynamically)
        strengths_df = all_stats_df.query('level == 0')[['node', 'y']].copy()
        strengths_df = (self.partition_df[[0, 'body', 'type', 'instance']]
                                .rename(columns={0: 'node'})
                                .merge(strengths_df, 'left', on=['node']))
        
        in_strengths_df = strengths_df.copy()
        out_strengths_df = strengths_df.copy()
        del strengths_df
        
        for df, color in [(in_strengths_df, 'red'),
                                    (out_strengths_df, 'green')]:
            df['color'] = color
            df['visible'] = False
            df['left'] = 0
            df['right'] = 0
            df['weight'] = 0
            df['height'] = 1.0
            df['rois'] = ''

        # Save members
        self.group_midpoints = group_midpoints
        self.all_stats_df = all_stats_df
        self.strengths_df = out_strengths_df
        self.in_strengths_df = in_strengths_df
        self.out_strengths_df = out_strengths_df
        

    def modify_doc(self, doc):
        
        group_midpoints = self.group_midpoints
        partition_df = self.partition_df
        all_stats_df = self.all_stats_df
        in_strengths_df = self.in_strengths_df
        out_strengths_df = self.out_strengths_df
        strong_connections_df = self.strong_connections_df
        
        BODY_TOOLTIPS = [
            ("body", "@body"),
            ("body_type", "@body_type"),
            ("body_instance", "@body_instance"),
        ]
    
        GROUP_TOOLTIPS = [
            ("most common", "@type_mode")
        ]
    
        body_hover = HoverTool(names=['body'], tooltips=BODY_TOOLTIPS)
        group_hover = HoverTool(names=['group', 'group_line'], tooltips=GROUP_TOOLTIPS, line_policy="prev")
    
        p = figure(title="block hierarchy",
                   x_axis_label='level',
                   y_axis_label='neuron',
                   plot_width=800,
                   plot_height=1000,
                   x_range=(-1.5, self.num_levels + 0.5),
                   output_backend="webgl",
                   tools=['ypan', 'ywheel_pan', 'ywheel_zoom', 'ybox_zoom', body_hover, group_hover, 'save', 'reset'],
                   active_drag='ybox_zoom',
                   active_scroll='ywheel_zoom')
    
        body_source_df = pd.DataFrame({
            'body_instance': [instance if instance else None for instance in partition_df['instance'].values],
            'body_pos': [group_midpoints[0][vertex] for vertex in partition_df.loc[:, 0].values],
            'body_type': [_type if _type else None for _type in partition_df['type'].values],
            'body': partition_df['body'].apply(str).values,
        })
    
        # Choose a pseudo-random hue for each type
        def choose_color_hsv(body_type):
            if not body_type:
                return (0.0, 0.0, 0.0)
            crc = zlib.crc32(body_type.encode('utf-8'))
            hue = (crc % 1024) / 1024
            return (hue, 1.0, 1.0)
        
        # Convert from HSV to RGB for display
        body_colors_hsv = np.array([*map(choose_color_hsv, body_source_df['body_type'].values)])
        body_colors_rgb = (255*hsv_to_rgb(body_colors_hsv)).astype(int)
        body_source_df['color'] = [*map(lambda c: f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}', body_colors_rgb)]
        body_source = ColumnDataSource(data=body_source_df)
    
        segment_source = ColumnDataSource(dict(
            x0=all_stats_df.query('level > 0')['x'].tolist(),
            x1=all_stats_df.query('level > 0')['parent_x'].tolist(),
            y0=all_stats_df.query('level > 0')['y'].tolist(),
            y1=all_stats_df.query('level > 0')['parent_y'].tolist(),
            color=all_stats_df.query('level > 0')['color'].values.tolist(),
            type_mode=all_stats_df.query('level > 0')['type_mode']
        ))
    
        # The leaf-level lines are drawn with a separate call, without tooltips
        # (to avoid overlapping with the body node tooltips).
        leaf_segment_source = ColumnDataSource(dict(
            x0=all_stats_df.query('level == 0')['x'].tolist(),
            x1=all_stats_df.query('level == 0')['parent_x'].tolist(),
            y0=all_stats_df.query('level == 0')['y'].tolist(),
            y1=all_stats_df.query('level == 0')['parent_y'].tolist(),
            color=all_stats_df.query('level == 0')['color'].values.tolist(),
        ))
        
        p.circle(0, 'body_pos', radius=0.4, radius_dimension='y', source=body_source, name='body', color='color')

        #group_source = ColumnDataSource(all_stats_df.query('level != 0'))
        #p.circle('x', 'y', radius=0.4, radius_dimension='y', source=group_source, name='group', color='color')
        
        p.segment(x0='x0', x1='x1', y0='y0', y1='y1', source=segment_source, line_width=1, name='group_line', color='color')
        p.segment(x0='x0', x1='x1', y0='y0', y1='y1', source=leaf_segment_source, line_width=1, name='leaf_line', color='color')
    
        # Fixme: would this look better as a LabelSet?
        #p.text(-0.3, body_pos, body_type, text_baseline="middle", text_align="right", text_font_size='10pt')
    
        IN_STRENGTH_TOOLTIPS = [
            ("from_body", "@body"),
            ("from_body_type", "@type"),
            ("from_body_instance", "@instance"),
            ("from_rois", "@rois")
        ]
        in_strength_hover = HoverTool(names=['in_strengths'], tooltips=IN_STRENGTH_TOOLTIPS)
    
        OUT_STRENGTH_TOOLTIPS = [
            ("to_body", "@body"),
            ("to_body_type", "@type"),
            ("to_body_instance", "@instance"),
            ("to_rois", "@rois")
        ]
        out_strength_hover = HoverTool(names=['out_strengths'], tooltips=OUT_STRENGTH_TOOLTIPS)
    
        # Strengths Plot
        sp = figure(title="connections",
                    plot_width=200,
                    plot_height=1000,
                    x_axis_label='log strength',
                    x_range=(-5, 5),
                    #x_axis_type="log",
                    y_range=p.y_range,
                    y_axis_label='neuron',
                    output_backend="webgl",
                    tools=['ypan', 'ywheel_zoom', 'ybox_zoom', in_strength_hover, out_strength_hover, 'save', 'reset'],
                    active_drag='ybox_zoom',
                    active_scroll='ywheel_zoom'
             )
    
    #     sp.toolbar.logo = None
    #     sp.toolbar_location = None
    
        in_strength_source = ColumnDataSource(in_strengths_df.iloc[0:0])
        out_strength_source = ColumnDataSource(out_strengths_df.iloc[0:0])
    
        sp.hbar(left='left', y='y', height='height', right='right', source=in_strength_source, line_width=1, name='in_strengths', color='color')
        sp.hbar(left='left', y='y', height='height', right='right', source=out_strength_source, line_width=1, name='out_strengths', color='color')
    
        def on_tap(event):
            _event_x, _event_y = event.x, event.y
    
            # Reset
            all_stats_df['color'] = 'gray'
    
            # Scale by the current plot dimensions before computing distance
            # FIXME: This assumes the plot is square.
            #        I should also compensate for the plot width/height in screen space.
            _display_width = p.x_range.end - p.x_range.start
            _display_height = p.y_range.end - p.y_range.start
            idx = all_stats_df.eval('((x - @_event_x)/@_display_width)**2 + ((y - @_event_y)/@_display_height)**2').idxmin()
            tap_level, node = all_stats_df.loc[idx, ['level', 'node']].values
            
            if tap_level == 0:
                child_df = all_stats_df.query('level == 0 and node == @node')
                clicked_body, clicked_type, clicked_instance = (partition_df
                                                                .rename(columns={0:'node'})
                                                                .query('node == @node')
                                                            [['body', 'type', 'instance']].iloc[0])
                print(f"You clicked body {clicked_body} ({clicked_type} / {clicked_instance})")
            else:
                print(f"You clicked level {tap_level}, node {node}")
    
            all_stats_df.loc[idx, 'color'] = 'cyan'
            _parents = {node}
            for _level in range(tap_level-1, -1, -1):
                child_df = all_stats_df.query('level == @_level and parent in @_parents')
                child_rows = child_df.index.values
                all_stats_df.loc[child_rows, 'color'] = 'cyan'
                _parents = {*all_stats_df.loc[child_rows, 'node']}
    
            leaf_segment_source.data['color'] = all_stats_df.query('level == 0')['color']
            #group_source.data['color'] = all_stats_df.query('level > 0')['color']
            segment_source.data['color'] = all_stats_df.query('level > 0')['color']
    
            # Final child_df after the above loop runs is the leaf (body) level
            assert (child_df['level'] == 0).all()
            _child_nodes = child_df['node'].values
            _child_bodies = partition_df.rename(columns={0: 'node'})[['node', 'body']].query('node in @_child_nodes')['body']
    
            # Reset
            in_strengths_df['visible'] = False
            in_strengths_df['left'] = 0
            in_strengths_df['right'] = 0
            in_strengths_df['weight'] = 0.0
    
            out_strengths_df['visible'] = False
            out_strengths_df['left'] = 0
            out_strengths_df['right'] = 0
            out_strengths_df['weight'] = 0.0
    
            strong_into_child_bodies = strong_connections_df.query('bodyId_post in @_child_bodies')
            _in_strengths_df = (in_strengths_df[['body']]
                                    .merge(strong_into_child_bodies, 'left', left_on='body', right_on='bodyId_pre'))
            in_strengths_df.loc[:, 'weight'] = _in_strengths_df['weight'].fillna(0.0)
            
            strong_children_outof_child_bodies = strong_connections_df.query('bodyId_pre in @_child_bodies')
            _out_strengths_df = (out_strengths_df[['body']]
                                    .merge(strong_children_outof_child_bodies, 'left', left_on='body', right_on='bodyId_post'))
            out_strengths_df.loc[:, 'weight'] = _out_strengths_df['weight'].fillna(0.0)
            
            in_rows = (in_strengths_df['weight'] > 0)
            in_strengths_df.loc[in_rows, 'visible'] = True
            in_strengths_df.loc[in_rows, 'left'] = -np.log10(in_strengths_df.loc[in_rows, 'weight'].values)
    
            out_rows = (out_strengths_df['weight'] > 0)
            out_strengths_df.loc[out_rows, 'visible'] = True
            out_strengths_df.loc[out_rows, 'right'] = np.log10(out_strengths_df.loc[out_rows, 'weight'].values)
    
            in_strength_source.data = in_strengths_df.query('left != 0 or right != 0')
            out_strength_source.data = out_strengths_df.query('left != 0 or right != 0')
    
        p.on_event(Tap, on_tap)
    
        #p.text(-0.1, body_pos, body_name, text_baseline="middle", text_align="right", text_font_size='10pt')
        #group_labels = labels = LabelSet(x='x', y='y', text='type_mode', level='overlay',
        #              x_offset=5, y_offset=5, angle=np.pi/4, source=group_source, render_mode='canvas')
        #p.add_layout(group_labels)
        
        doc.add_root(bl.row(sp, p))
