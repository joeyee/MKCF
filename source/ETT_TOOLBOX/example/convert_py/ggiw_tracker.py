#cell 0
import bokeh.plotting as bk
from bokeh.io import output_notebook
output_notebook()

#cell 1
import sys
sys.path.append('../') # go to parent dir
import numpy as np
np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
np.seterr('warn')
import os
import cProfile, pstats, io

#cell 2
from bokeh.palettes import Category20
colors = Category20[20]

#cell 3
### Load Data

#cell 4
path = os.path.join(os.getcwd(), 'data')
frame_start, frame_end = -1, 100
measurements = np.load(os.path.join(path, 'simulated_data' + '.npy'))
gt_bboxes = np.load(os.path.join(path, 'gt_bboxes' + '.npy'))
if frame_end is -1:
    measurements = measurements[measurements['ts'] >= frame_start]
    gt_bboxes = gt_bboxes[gt_bboxes['ts'] >= frame_start]
else:
    measurements = measurements[ (measurements['ts'] >= frame_start) & (measurements['ts'] <= frame_end)]
    gt_bboxes = gt_bboxes[ (gt_bboxes['ts'] >= frame_start) & (gt_bboxes['ts'] <= frame_end)]

#cell 5
## Gamma Gaussian Inverse Wishart Tracker

#cell 6
### Create Tracker

#cell 7
steps = max(measurements['ts']) + 1

# tracker config
dt = 0.04
sa_sq = 1.5 ** 2

config = {
    'steps': steps + 1,
    'd': 2,
    's': 2,
    'sd': 4,
    'lambda_d': 0.999,
    'eta': 1.0 / 0.999,
    'n_factor': 1.0,
    'f': np.asarray([[1, dt], [0, 1]], dtype='f4'),
    'q': sa_sq * np.outer(np.asarray([dt ** 2 / 2.0, dt]), np.asarray([dt ** 2 / 2.0, dt])),
    'h': np.asarray([[1, 0]], dtype='f4'),
    'init_m': [6.5, 2.5, 12, 0],
    'init_c': 2 ** 2 * sa_sq * np.outer(np.asarray([dt ** 2 / 2.0, dt]), np.asarray([dt ** 2 / 2.0, dt])),
    'init_alpha': 5,
    'init_beta': 1,
    'init_v': np.diag([4, 2]),
    'init_nu': 5,
}

from models import GgiwTracker as Tracker

#cell 8
### Run Tracker

#cell 9
# tracker definition
tracker = Tracker(dt=dt, **config)

pr = cProfile.Profile()
for i in range(steps):
    scan = measurements[measurements['ts'] == i]
    pr.enable()
    tracker.step(scan)
    pr.disable()

estimates, log_lik = tracker.extract()
bboxes = tracker.extrackt_bbox()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(10)

print('total time: {:f}, total steps: {:d}, average step time: {:f}'.format(
    ps.total_tt, max(measurements['ts']) - 2, ps.total_tt / (max(measurements['ts']) - 2)))
print(s.getvalue())

#cell 10
### Calculate Wasserstein Metric

#cell 11
from metric import point_set_wasserstein_distance as pt_wsd
from misc import convert_rectangle_to_eight_point
eight_pts = convert_rectangle_to_eight_point(bboxes[1:])  # drop prior bounding box
eight_pts_gt = convert_rectangle_to_eight_point(gt_bboxes)

wsd = np.zeros(len(gt_bboxes), dtype='f8')
for i, (pts, gt_pts) in enumerate(zip(eight_pts, eight_pts_gt)):
    wsd[i] = pt_wsd(pts, gt_pts, p=2.0)

#cell 12
### data source version

#cell 13
from bokeh.models import ColumnDataSource

stride = 5

est_source = ColumnDataSource(
        data={
            'ts': estimates['ts'],
            'm_x': estimates['m'][:, 0],
            'm_y': estimates['m'][:, 1],
            'v_x': estimates['m'][:, 2],
            'v_y': estimates['m'][:, 3],
            'nu': estimates['nu'],
            'v': estimates['v'],
            'v_hat': estimates['v_hat'],
            'phi': 0.5 * np.arctan2(2 * estimates['v_hat'][:, 0, 1], estimates['v_hat'][:, 0, 0] - estimates['v_hat'][:, 1, 1]),
            'loglik': log_lik,
        }
    )

meas_source = ColumnDataSource(
        data={
            'ts': measurements['ts'],
            'x': measurements['xy'][:, 0],
            'y': measurements['xy'][:, 1],
        }
)

sel = measurements['ts'] % stride == 0
meas_source_sel = ColumnDataSource(
        data={
            'ts': measurements['ts'][sel],
            'x': measurements['xy'][sel, 0],
            'y': measurements['xy'][sel, 1],
        }
)

sel = bboxes['ts'] % stride == 0
bbox_source = ColumnDataSource(
        data={
            'ts': bboxes['ts'][sel],
            'm_x': bboxes['center_xy'][sel, 0],
            'm_y': bboxes['center_xy'][sel, 1],
            'phi': bboxes['orientation'][sel],
            'l': bboxes['dimension'][sel, 0],
            'w': bboxes['dimension'][sel, 1],
        }
)

gt_bbox_source = ColumnDataSource(
        data={
            'ts': gt_bboxes['ts'],
            'm_x': gt_bboxes['center_xy'][:, 0],
            'm_y': gt_bboxes['center_xy'][:, 1],
            'phi': gt_bboxes['orientation'],
            'l': gt_bboxes['dimension'][:, 0],
            'w': gt_bboxes['dimension'][:, 1],
            'wsd': wsd,
        }
)

sel = gt_bboxes['ts'] % stride == 0
gt_bbox_source_sel = ColumnDataSource(
        data={
            'ts': gt_bboxes['ts'][sel],
            'm_x': gt_bboxes['center_xy'][sel, 0],
            'm_y': gt_bboxes['center_xy'][sel, 1],
            'phi': gt_bboxes['orientation'][sel],
            'l': gt_bboxes['dimension'][sel, 0],
            'w': gt_bboxes['dimension'][sel, 1],
        }
)

#cell 14
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, BoxSelectTool, PanTool, BoxZoomTool, WheelZoomTool, UndoTool, RedoTool, ResetTool, SaveTool
from numpy.linalg import eigvals

TOOLS = [PanTool(), BoxSelectTool(), BoxZoomTool(), WheelZoomTool(), UndoTool(), RedoTool(), ResetTool(), SaveTool()]

top = figure(tools=TOOLS, width=800, height=350, title=None, match_aspect=True, output_backend="webgl")
meas_scatter = top.circle('x', 'y', color='#303030', legend='measurements', size=1, alpha=0.2, source=meas_source)
track_plot = top.line('m_x', 'm_y', legend='track', source=est_source)
hover = HoverTool(renderers=[track_plot],
        tooltips=[
            ("index", "$index"),
            (r"(x, y, phi, v)", "(@m_x, @m_y, @phi, @v)"),
            ("(length, width)", "(@d_x, @d_y)"),
        ]
    )
top.add_tools(hover)

bottom = figure(tools=TOOLS, width=800, height=350, title=None, 
                x_range=top.x_range, y_range=top.y_range, output_backend="webgl")
bottom.circle('x', 'y', color='#303030', legend='measurements', size=1, alpha=0.2, source=meas_source_sel)
bottom.rect(x='m_x', y='m_y', width='l', height='w', angle='phi', color="#CAB2D6", 
            fill_alpha=0.2, source=bbox_source, legend='bounding boxes')
bottom.rect(x='m_x', y='m_y', width='l', height='w', angle='phi', color='#98df8a', 
            fill_alpha=0.2, source=gt_bbox_source_sel, legend='ground truth boxes')

sel = estimates['ts'] % stride == 0
wh = eigvals(estimates['v_hat'][sel]).T
wh = np.sqrt(wh)
bottom.ellipse(estimates['m'][sel, 0], estimates['m'][sel, 1], width=3 * wh[0], height=3 * wh[1], 
               angle=0.5 * np.arctan2(2 * estimates['v_hat'][sel, 0, 1], estimates['v_hat'][sel, 0, 0] - estimates['v_hat'][sel, 1, 1]), 
               color=colors[0], fill_alpha=0.2, legend='shape')

bottom.legend.click_policy="hide"

p = gridplot([[top], [bottom]])
show(p)

#cell 15
### Statistics

#cell 16
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import gridplot, column

f_wsd = figure(tools=TOOLS, width=800, height=350, title='Wasserstein Distance')
f_log_lik = figure(tools=TOOLS, width=800, height=350, title='Likelihood', x_range=f_wsd.x_range)
f_wsd.line('ts', 'wsd', legend='wasserstein distance', source=gt_bbox_source, color=colors[0])
f_log_lik.line('ts', 'loglik', legend='log likelihood', source=est_source, color=colors[0])
pan1 = Panel(child=column([f_wsd, f_log_lik]), title="Likelihood and Wasserstein Distance")

f_position = figure(tools=TOOLS, width=800, height=350, title='Position')
f_orientation = figure(tools=TOOLS, width=800, height=350, title='Orientation', x_range=f_position.x_range)
f_velocity = figure(tools=TOOLS, width=800, height=350, title='Velocity', x_range=f_position.x_range)
f_dimension = figure(tools=TOOLS, width=800, height=350, title='Dimension', x_range=f_position.x_range)
f_position.line('ts', 'm_x', legend='x position', source=est_source, color=colors[0])
f_position.line('ts', 'm_y', legend='y position', source=est_source, color=colors[2])
f_position.line('ts', 'm_x', legend='ground truth x position', source=gt_bbox_source, color=colors[1])
f_position.line('ts', 'm_y', legend='ground truth y position', source=gt_bbox_source, color=colors[3])
f_orientation.line('ts', 'phi', legend='orientation', source=est_source, color=colors[0])
f_orientation.line('ts', 'phi', legend='ground truth orientation', source=gt_bbox_source, color=colors[1])
f_velocity.line('ts', 'v_x', legend='x velocity', source=est_source, color=colors[0])
f_velocity.line('ts', 'v_y', legend='y velocity', source=est_source, color=colors[2])
# f_velocity.line('ts', 'v', legend='ground truth orientation', source=gt_bbox_source, color=colors[3])
f_dimension.line('ts', 'l', legend='x dimension scaling', source=bbox_source, color=colors[0])
f_dimension.line('ts', 'w', legend='y dimension scaling', source=bbox_source, color=colors[2])
f_dimension.line('ts', 'l', legend='ground truth x dimension', source=gt_bbox_source, color=colors[1])
f_dimension.line('ts', 'w', legend='ground truth y dimension', source=gt_bbox_source, color=colors[3])
pan2 = Panel(child=column([f_position, f_orientation, f_velocity, f_dimension]), title="Kinematic and Extent")

for f in [f_wsd, f_log_lik, f_position, f_orientation, f_velocity, f_dimension]:
    f.legend.click_policy="mute"

tabs = Tabs(tabs=[pan1, pan2])
show(tabs)

#cell 17


