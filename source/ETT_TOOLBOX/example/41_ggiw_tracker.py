"""
Convert the 40_ggiw_tracker.ipynb to 41_ggiw.py and drawing the rectangles of the estimations.
"""
import sys
#add ETT_TOOLBOX to the working directory for 'from models import spline'
sys.path.insert(0,'/Users/yizhou/code/MKCF_MAC_Python3.6/source/ETT_TOOLBOX/')
import numpy as np
np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
np.seterr('warn')
import os
import cProfile, pstats, io
import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


#loading data
path = os.path.join(os.getcwd(), 'data')
frame_start, frame_end = -1, 100
measurements = np.load(os.path.join(path, 'simulated_data' + '.npy'))
gt_bboxes    = np.load(os.path.join(path, 'gt_bboxes' + '.npy'))

if frame_end is -1:
    measurements = measurements[measurements['ts'] >= frame_start]
    gt_bboxes    = gt_bboxes[gt_bboxes['ts'] >= frame_start]
else:
    measurements = measurements[ (measurements['ts'] >= frame_start) & (measurements['ts'] <= frame_end)]
    gt_bboxes    = gt_bboxes[ (gt_bboxes['ts'] >= frame_start) & (gt_bboxes['ts'] <= frame_end)]

steps = max(measurements['ts']) + 1

# tracker config
dt = 0.04           #step time difference
sa_sq = 1.5 ** 2

config = {
    'steps': steps + 1,       # total time steps
    'd': 2,                   # dimensions
    's': 2,                   #
    'sd': 4,                  #
    'lambda_d': 0.999,        # exp(-dt/tal)
    'eta': 1.0 / 0.999,       # factor for alpha iterative decrease
    'n_factor': 1.0,          #
    'f': np.asarray([[1, dt], [0, 1]], dtype='f4'),    # sub-matrix of translation matrix
    'q': sa_sq * np.outer(np.asarray([dt ** 2 / 2.0, dt]), np.asarray([dt ** 2 / 2.0, dt])),  # transition  noise
    'h': np.asarray([[1, 0]], dtype='f4'),             # sub measure matrix
    'init_m': [6.5, 2.5, 12, 0],                       # initial state mean (cx, cy, vx, vy)
    'init_c': 2 ** 2 * sa_sq * np.outer(np.asarray([dt ** 2 / 2.0, dt]), np.asarray([dt ** 2 / 2.0, dt])), #initial state sub-covariance
    'init_alpha': 5, # Gammar distribution parameter alpha
    'init_beta': 1,  # gammar distribution parameter beta
    'init_v': np.diag([4, 2]), #shape matrix in iw distribution
    'init_nu': 5,              #free degree in iw_distribution
}

from models import GgiwTracker as Tracker
# tracker definition
tracker = Tracker(dt=dt, **config)

#Profile the time consuming of the tracker step by step!
pr = cProfile.Profile()
for i in range(steps):
    scan = measurements[measurements['ts'] == i]
    pr.enable()
    tracker.step(scan)
    pr.disable()

estimates, log_lik = tracker.extract()
bboxes = tracker.extrackt_bbox()

## print the profile!
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats(10)
#
# print('total time: {:f}, total steps: {:d}, average step time: {:f}'.format(
#     ps.total_tt, max(measurements['ts']) - 2, ps.total_tt / (max(measurements['ts']) - 2)))
# print(s.getvalue())

from matplotlib.patches import Ellipse
import numpy.linalg as la
import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
stride = 10
# d = estimates['V_hat'].shape[-1]
ell_s_factor = 3.0
rad_to_deg = 180.0 / np.pi

for i in np.arange(0, len(estimates), stride):
    for est in estimates:
        if est['ts'] % stride == 0:
            w, v = la.eig(est['v_hat'])
            angle_deg = np.arctan2(v[1, 0], v[0, 0])
            angle_deg *= rad_to_deg
            e = Ellipse(xy=est['m'], width=w[0], height=w[1], angle=angle_deg, alpha=0.5, color='#1f77b4')
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor('none')
            e.set_edgecolor('#1f77b4')
            #ax.add_artist(e)

plt.xlim(0, 75)
plt.ylim(0, 20)
plt.show(block=True)





# stride = 5
#
# color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
#                   '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
#                   '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
#                   '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
#
# plt.style.use('ggplot')
# fig, ax = plt.subplots(2, 1, figsize=(20.0, 10.0))
# fig.suptitle('Single Target Framework Data Generation', fontsize=12, x=0.02, horizontalalignment='left')
#
# for a in ax:
#     a.set_xlabel(r'$x$')
#     a.set_ylabel(r'$y$')
#     a.set_aspect('equal')
#
# ax[1].get_shared_x_axes().join(ax[0], ax[1])
# ax[1].get_shared_y_axes().join(ax[0], ax[1])
#
#
# def plot_rectangle(bboxes, ax, step, c='#ff7f0e'):
#     for bbox in bboxes[bboxes['ts'] % step == 0]:
#         s_phi_offset, c_phi_offset = np.sin(bbox['orientation']), np.cos(bbox['orientation'])
#         rot = np.array([[c_phi_offset, - s_phi_offset], [s_phi_offset, c_phi_offset]])
#         offset_xy = np.dot(rot, 0.5 * bbox['dimension'])
#
#         r = Rectangle(xy=bbox['center_xy'] - offset_xy, width=bbox['dimension'][0], height=bbox['dimension'][1],
#                       angle=np.rad2deg(bbox['orientation']))
#
#         ax.add_artist(r)
#         r.set_clip_box(ax.bbox)
#         r.set_alpha(0.8)
#         r.set_facecolor('none')
#         r.set_edgecolor(c)
#
#
# plot_rectangle(bboxes, ax[1], stride, c='#ff7f0e')
# #ax[0].plot(gt_bboxes['m'][:, 0], gt_bboxes['m'][:, 1], label='track', c=color_sequence[0])
#
# sel = measurements['ts'] % stride == 0
# ax[1].plot(measurements['xy'][sel, 0], measurements['xy'][sel, 1],
#            c='k', marker='.', linewidth=0, markersize=0.5, alpha=0.5, label='measurements')
#
# ax[0].plot(measurements['xy'][:, 0], measurements['xy'][:, 1],
#            c='k', marker='.', linewidth=0, markersize=0.5, alpha=0.5, label='measurements')
#
# for a in ax:
#     a.legend()
#
# plt.show()


# from metric import point_set_wasserstein_distance as pt_wsd
# from misc import convert_rectangle_to_eight_point
# eight_pts = convert_rectangle_to_eight_point(bboxes[1:])  # drop prior bounding box
# eight_pts_gt = convert_rectangle_to_eight_point(gt_bboxes)
#
# wsd = np.zeros(len(gt_bboxes), dtype='f8')
# for i, (pts, gt_pts) in enumerate(zip(eight_pts, eight_pts_gt)):
#     wsd[i] = pt_wsd(pts, gt_pts, p=2.0)
#
#
# from bokeh.models import ColumnDataSource
#
# stride = 5
#
# est_source = ColumnDataSource(
#         data={
#             'ts': estimates['ts'],
#             'm_x': estimates['m'][:, 0],
#             'm_y': estimates['m'][:, 1],
#             'v_x': estimates['m'][:, 2],
#             'v_y': estimates['m'][:, 3],
#             'nu': estimates['nu'],
#             'v': estimates['v'],
#             'v_hat': estimates['v_hat'],
#             'phi': 0.5 * np.arctan2(2 * estimates['v_hat'][:, 0, 1], estimates['v_hat'][:, 0, 0] - estimates['v_hat'][:, 1, 1]),
#             'loglik': log_lik,
#         }
#     )
#
# meas_source = ColumnDataSource(
#         data={
#             'ts': measurements['ts'],
#             'x': measurements['xy'][:, 0],
#             'y': measurements['xy'][:, 1],
#         }
# )
#
# sel = measurements['ts'] % stride == 0
# meas_source_sel = ColumnDataSource(
#         data={
#             'ts': measurements['ts'][sel],
#             'x': measurements['xy'][sel, 0],
#             'y': measurements['xy'][sel, 1],
#         }
# )
#
# sel = bboxes['ts'] % stride == 0
# bbox_source = ColumnDataSource(
#         data={
#             'ts': bboxes['ts'][sel],
#             'm_x': bboxes['center_xy'][sel, 0],
#             'm_y': bboxes['center_xy'][sel, 1],
#             'phi': bboxes['orientation'][sel],
#             'l': bboxes['dimension'][sel, 0],
#             'w': bboxes['dimension'][sel, 1],
#         }
# )
#
# gt_bbox_source = ColumnDataSource(
#         data={
#             'ts': gt_bboxes['ts'],
#             'm_x': gt_bboxes['center_xy'][:, 0],
#             'm_y': gt_bboxes['center_xy'][:, 1],
#             'phi': gt_bboxes['orientation'],
#             'l': gt_bboxes['dimension'][:, 0],
#             'w': gt_bboxes['dimension'][:, 1],
#             'wsd': wsd,
#         }
# )
#
# sel = gt_bboxes['ts'] % stride == 0
# gt_bbox_source_sel = ColumnDataSource(
#         data={
#             'ts': gt_bboxes['ts'][sel],
#             'm_x': gt_bboxes['center_xy'][sel, 0],
#             'm_y': gt_bboxes['center_xy'][sel, 1],
#             'phi': gt_bboxes['orientation'][sel],
#             'l': gt_bboxes['dimension'][sel, 0],
#             'w': gt_bboxes['dimension'][sel, 1],
#         }
# )
# from bokeh.palettes import Category20
# colors = Category20[20]
#
# from bokeh.layouts import gridplot
# from bokeh.plotting import figure, output_file, show
# from bokeh.models import HoverTool, BoxSelectTool, PanTool, BoxZoomTool, WheelZoomTool, UndoTool, RedoTool, ResetTool, SaveTool
# from numpy.linalg import eigvals
#
# TOOLS = [PanTool(), BoxSelectTool(), BoxZoomTool(), WheelZoomTool(), UndoTool(), RedoTool(), ResetTool(), SaveTool()]
#
# top = figure(tools=TOOLS, width=800, height=350, title=None, match_aspect=True, output_backend="webgl")
# meas_scatter = top.circle('x', 'y', color='#303030', legend='measurements', size=1, alpha=0.2, source=meas_source)
# track_plot = top.line('m_x', 'm_y', legend='track', source=est_source)
# hover = HoverTool(renderers=[track_plot],
#         tooltips=[
#             ("index", "$index"),
#             (r"(x, y, phi, v)", "(@m_x, @m_y, @phi, @v)"),
#             ("(length, width)", "(@d_x, @d_y)"),
#         ]
#     )
# top.add_tools(hover)
#
# bottom = figure(tools=TOOLS, width=800, height=350, title=None,
#                 x_range=top.x_range, y_range=top.y_range, output_backend="webgl")
# bottom.circle('x', 'y', color='#303030', legend='measurements', size=1, alpha=0.2, source=meas_source_sel)
# bottom.rect(x='m_x', y='m_y', width='l', height='w', angle='phi', color="#CAB2D6",
#             fill_alpha=0.2, source=bbox_source, legend='bounding boxes')
# bottom.rect(x='m_x', y='m_y', width='l', height='w', angle='phi', color='#98df8a',
#             fill_alpha=0.2, source=gt_bbox_source_sel, legend='ground truth boxes')
#
# sel = estimates['ts'] % stride == 0
# wh = eigvals(estimates['v_hat'][sel]).T
# wh = np.sqrt(wh)
# bottom.ellipse(estimates['m'][sel, 0], estimates['m'][sel, 1], width=3 * wh[0], height=3 * wh[1],
#                angle=0.5 * np.arctan2(2 * estimates['v_hat'][sel, 0, 1], estimates['v_hat'][sel, 0, 0] - estimates['v_hat'][sel, 1, 1]),
#                color=colors[0], fill_alpha=0.2, legend='shape')
#
# bottom.legend.click_policy="hide"
#
# p = gridplot([[top], [bottom]])
# show(p)
#
# from bokeh.models.widgets import Panel, Tabs
# from bokeh.layouts import gridplot, column
#
# f_wsd = figure(tools=TOOLS, width=800, height=350, title='Wasserstein Distance')
# f_log_lik = figure(tools=TOOLS, width=800, height=350, title='Likelihood', x_range=f_wsd.x_range)
# f_wsd.line('ts', 'wsd', legend='wasserstein distance', source=gt_bbox_source, color=colors[0])
# f_log_lik.line('ts', 'loglik', legend='log likelihood', source=est_source, color=colors[0])
# pan1 = Panel(child=column([f_wsd, f_log_lik]), title="Likelihood and Wasserstein Distance")
#
# f_position = figure(tools=TOOLS, width=800, height=350, title='Position')
# f_orientation = figure(tools=TOOLS, width=800, height=350, title='Orientation', x_range=f_position.x_range)
# f_velocity = figure(tools=TOOLS, width=800, height=350, title='Velocity', x_range=f_position.x_range)
# f_dimension = figure(tools=TOOLS, width=800, height=350, title='Dimension', x_range=f_position.x_range)
# f_position.line('ts', 'm_x', legend='x position', source=est_source, color=colors[0])
# f_position.line('ts', 'm_y', legend='y position', source=est_source, color=colors[2])
# f_position.line('ts', 'm_x', legend='ground truth x position', source=gt_bbox_source, color=colors[1])
# f_position.line('ts', 'm_y', legend='ground truth y position', source=gt_bbox_source, color=colors[3])
# f_orientation.line('ts', 'phi', legend='orientation', source=est_source, color=colors[0])
# f_orientation.line('ts', 'phi', legend='ground truth orientation', source=gt_bbox_source, color=colors[1])
# f_velocity.line('ts', 'v_x', legend='x velocity', source=est_source, color=colors[0])
# f_velocity.line('ts', 'v_y', legend='y velocity', source=est_source, color=colors[2])
# # f_velocity.line('ts', 'v', legend='ground truth orientation', source=gt_bbox_source, color=colors[3])
# f_dimension.line('ts', 'l', legend='x dimension scaling', source=bbox_source, color=colors[0])
# f_dimension.line('ts', 'w', legend='y dimension scaling', source=bbox_source, color=colors[2])
# f_dimension.line('ts', 'l', legend='ground truth x dimension', source=gt_bbox_source, color=colors[1])
# f_dimension.line('ts', 'w', legend='ground truth y dimension', source=gt_bbox_source, color=colors[3])
# pan2 = Panel(child=column([f_position, f_orientation, f_velocity, f_dimension]), title="Kinematic and Extent")
#
# for f in [f_wsd, f_log_lik, f_position, f_orientation, f_velocity, f_dimension]:
#     f.legend.click_policy="mute"
#
# tabs = Tabs(tabs=[pan1, pan2])
# show(tabs)