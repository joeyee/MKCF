#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of KCF tracker

import cv2
import numpy as np

import ast

import pylab

#TODO list on 2017-10-17 对KCF的改进
#TODO 考虑根据K的高斯朝向来修改cos_window以及分类器y，同时根据cos_window的分布来调整跟踪目标的尺度。
#TODO 本周五引入KalmanFilter做跟踪比较，需要引入目标轮廓定位的方法，使之作为KF滤波的输入。波门。(周五花了一天来学习GMM，落下了功课)


debug = False

class KCFTracker(object):
    """
    define the tracker model, inclding init(), update() submodels
    """
    # def __init__(self):
    #     return

    #init function is different for varied trackers
    def init(self, img, rect ):
        im_width = img.shape[1]
        im_heihgt= img.shape[0]
        ys = pylab.floor(rect[1]) + pylab.arange(rect[3], dtype=int)
        xs = pylab.floor(rect[0]) + pylab.arange(rect[2], dtype=int)
        ys = ys.astype(int)
        xs = xs.astype(int)
        # check for out-of-bounds coordinates,
        # and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= img.shape[0]] = img.shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        roi = self.get_imageROI(img, rect)

        self.init_frame = img.copy()
        self.canvas     = img.copy()
        #pos is the center postion of the tracking object (cy,cx)
        pos = pylab.array([rect[1] + rect[3]/2, rect[0] + rect[2]/2])
        self.pos_list   = [pos]
        self.roi_list   = [roi]
        self.rect_list  = [rect]
        self.trackNo    = 0
        # parameters according to the paper --

        padding = 1.0  # extra area surrounding the target(扩大窗口的因子，默认扩大2倍)
        # spatial bandwidth (proportional to target)
        output_sigma_factor = 1 / float(16)
        self.sigma = 0.2  # gaussian kernel bandwidth
        self.lambda_value = 1e-2  # regularization
        # linear interpolation factor for adaptation
        self.interpolation_factor = 0.075


        #target_ze equals to [rect3, rect2]
        target_sz = pylab.array([int(rect[3]), int(rect[2])])
        # window size(Extended window size), taking padding into account
        window_sz = pylab.floor(target_sz * (1 + padding))

        self.window_sz = window_sz
        self.target_sz = target_sz

        # desired output (gaussian shaped), bandwidth proportional to target size
        output_sigma = pylab.sqrt(pylab.prod(target_sz)) * output_sigma_factor

        grid_y = pylab.arange(window_sz[0]) - pylab.floor(window_sz[0] / 2)
        grid_x = pylab.arange(window_sz[1]) - pylab.floor(window_sz[1] / 2)
        # [rs, cs] = ndgrid(grid_x, grid_y)
        rs, cs = pylab.meshgrid(grid_x, grid_y)
        y = pylab.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf= pylab.fft2(y)
        # store pre-computed cosine window
        self.cos_window = pylab.outer(pylab.hanning(window_sz[0]), pylab.hanning(window_sz[1]))


        # get subwindow at current estimated target position, to train classifer
        x = self.get_subwindow(img, pos, window_sz, self.cos_window)
        # Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
        k = self.dense_gauss_kernel(self.sigma, x)
        #storing computed alphaf and z for next frame iteration
        self.alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))  # Eq. 7
        self.z = x
        #return initialization status
        return True

    def update(self, new_img):
        self.canvas   = new_img.copy()
        self.trackNo +=1

        # get subwindow at current estimated target position, to train classifer
        x = self.get_subwindow(new_img, self.pos_list[-1], self.window_sz, self.cos_window)
        # calculate response of the classifier at all locations
        k = self.dense_gauss_kernel(self.sigma, x, self.z)
        kf = pylab.fft2(k)
        alphaf_kf = pylab.multiply(self.alphaf, kf)
        response = pylab.real(pylab.ifft2(alphaf_kf))  # Eq. 9

        # target location is at the maximum response
        r = response
        row, col = pylab.unravel_index(r.argmax(), r.shape)
        #roi rect's topleft point add [row, col]
        pos = self.pos_list[-1] - pylab.floor(self.window_sz / 2) + [row, col]
        rect = pylab.array([pos[1]- self.target_sz[1]/2, pos[0] - self.target_sz[0]/2, self.target_sz[1], self.target_sz[0]])
        rect = rect.astype(np.int)

        if debug:
            if self.trackNo == 1:
                #pylab.ion()  # interactive mode on
                self.fig, self.axes = pylab.subplots(ncols=3)
                self.fig.show()
                # We need to draw the canvas before we start animating...
                self.fig.canvas.draw()

                k_img = self.axes[0].imshow(k,animated=True)
                x_img = self.axes[1].imshow(x,animated=True)
                r_img = self.axes[2].imshow(response,animated=True)

                self.subimgs = [k_img, x_img, r_img]
                # Let's capture the background of the figure
                self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]

                # tracking_rectangle = pylab.Rectangle((0, 0), 0, 0)
                # tracking_rectangle.set_color((1, 0, 0, 0.5))
                # tracking_figure_axes.add_patch(tracking_rectangle)
                #
                # gt_point = pylab.Circle((0, 0), radius=5)
                # gt_point.set_color((0, 0, 1, 0.5))
                # tracking_figure_axes.add_patch(gt_point)
                # tracking_figure_title = tracking_figure.suptitle("")
                pylab.show(block=False)
                #self.fig.show()
            else:
                self.subimgs[0].set_data(k)
                self.subimgs[1].set_data(x)
                self.subimgs[2].set_data(response)
                items = enumerate(zip(self.subimgs, self.axes, self.backgrounds), start=1)
                for j, (subimg, ax, background) in items:
                    self.fig.canvas.restore_region(background)
                    ax.draw_artist(subimg)
                    self.fig.canvas.blit(ax.bbox)
                pylab.show(block=False)

        #computing new_alphaf and observed x as z
        x = self.get_subwindow(new_img, pos, self.window_sz, self.cos_window)
        # Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
        k = self.dense_gauss_kernel(self.sigma, x)
        new_alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))  # Eq. 7
        new_z = x

        # subsequent frames, interpolate model
        f = self.interpolation_factor
        self.alphaf = (1 - f) * self.alphaf + f * new_alphaf
        self.z = (1 - f) * self.z + f * new_z


        self.roi_list.append(self.get_imageROI(new_img, rect))
        self.pos_list.append(pos)
        self.rect_list.append(rect)
        ok = 1
        return ok, rect

    def show_precision(self, tt_file, gt_file):
        #TT_file is for tracked_target file
        #GT_file is for ground_truth file
        with open(gt_file,'r') as gtobj:
            gt_lines = gtobj.readlines()
        with open(tt_file, 'r') as ttobj:
            tt_lines = ttobj.readlines()

        tt_pos_list = []
        gt_pos_list = []

        #tt and gt gets same structure, each line is one key dict, the key is frame + frame_id
        # in each line's dict, the value of the key is also a dict, contains 'boundingbox', 'center', etc.
        #{'frame id':{'Boundingbox:{(x y w h)}, 'Center':{cx, cy}}, Format: a dict contains a dict
        #if the frame id contains no gt or tt info, then the son-dict is empty, as {'frame id':{}}
        #only the not empty tt_dict and gt_dict is countered in the precision measure.

        tt_dict = {}
        gt_dict = {}
        for ttline in tt_lines:
            tt = ast.literal_eval(ttline)
            tt_dict.update(tt)
        for gtline in gt_lines:
            gt = ast.literal_eval(gtline)
            gt_dict.update(gt)

        #method 1, the dict is not follow the sequential frame id, but it is ok for precision computing
        for ttkey in tt_dict:
            if ttkey not in gt_dict:
                continue
            tt_info_dict = tt_dict[ttkey]
            gt_info_dict = gt_dict[ttkey]
            if tt_info_dict != {} and gt_info_dict != {}:
                tt_pos = tt_info_dict['Center']
                gt_pos = gt_info_dict['Center']
                gt_pos_list.append(gt_pos)
                tt_pos_list.append(tt_pos)

        framestr = tt.keys()[0]
        tt_dict[framestr] = tt[framestr]

        print ('compared tracking numbers %d \n' % len(tt_pos_list))

        gt_poses = np.array(gt_pos_list, dtype = float)
        tt_poses = np.array(tt_pos_list, dtype = float)

        dist = np.sqrt( (gt_poses[:,0] - tt_poses[:,0])**2 + (gt_poses[:,1] - tt_poses[:,1])**2 )
        precision = np.zeros((100,1),np.float)
        for thresh in range(100):
            tracked_nums = np.sum(dist <= thresh, dtype = float)
            precision[thresh] = tracked_nums/len(dist)

        #the X_axis is the threashold and the y_axis is the precision.
        pylab.figure()
        pylab.plot(precision)
        #pylab.show()

        #method 2, the list follow the sequential frame numbers.
        tt_pos_list = []
        gt_pos_list = []
        frame_id_list = []
        #for sequential observe each position, the x_axis is the frame_id, y_axis is the distance
        for ttline in tt_lines:
            tt = ast.literal_eval(ttline)
            frameid_str = tt.keys()[0]
            frame_id    = frameid_str.split(' ')[1]
            if frameid_str not in gt_dict:
                continue
            else:
                if gt_dict[frameid_str] != {}:
                    tt_pos_list.append(tt_dict[frameid_str]['Center'])
                    gt_pos_list.append(gt_dict[frameid_str]['Center'])
                    frame_id_list.append(int(frame_id))

        gt_seq_poses = np.array(gt_pos_list, dtype = float)
        tt_seq_poses = np.array(tt_pos_list, dtype = float)
        seq_dist     = np.sqrt( (gt_seq_poses[:,0] - tt_seq_poses[:,0])**2 + (gt_seq_poses[:,1] - tt_seq_poses[:,1])**2 )
        pylab.figure()
        pylab.plot(frame_id_list[:150], seq_dist[:150])
        pylab.show()

    def get_imageROI(self, im, rect):

        ys = pylab.floor(rect[1])  + pylab.arange(rect[3], dtype=int)
        xs = pylab.floor(rect[0])  + pylab.arange(rect[2], dtype=int)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates,
        # and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1
        roi = im[pylab.ix_(ys, xs)]
        return roi

    def get_subwindow(self, im, pos, sz, cos_window):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """

        if pylab.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = pylab.floor(pos[0]) \
            + pylab.arange(sz[0], dtype=int) - pylab.floor(sz[0]/2)
        xs = pylab.floor(pos[1]) \
            + pylab.arange(sz[1], dtype=int) - pylab.floor(sz[1]/2)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates,
        # and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1
        #zs = range(im.shape[2])

        # extract image
        #out = im[pylab.ix_(ys, xs, zs)]
        out = im[pylab.ix_(ys, xs)]

        if debug:
            print("Out max/min value==", out.max(), "/", out.min())
            pylab.figure()
            pylab.imshow(out, cmap=pylab.cm.gray)
            pylab.title("cropped subwindow")

        #pre-process window --
        # normalize to range -0.5 .. 0.5
        # pixels are already in range 0 to 1
        out = out.astype(pylab.float64) - 0.5

        # apply cosine window
        out = pylab.multiply(cos_window, out)

        return out


    def dense_gauss_kernel(slef, sigma, x, y=None):
        """
        Gaussian Kernel with dense sampling.
        Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
        between input images X and Y, which must both be MxN. They must also
        be periodic (ie., pre-processed with a cosine window). The result is
        an MxN map of responses.

        If X and Y are the same, ommit the third parameter to re-use some
        values, which is faster.
        """
        xf = pylab.fft2(x)  # x in Fourier domain
        x_flat = x.flatten()
        xx = pylab.dot(x_flat.transpose(), x_flat)  # squared norm of x

        if y is not None:
            # general case, x and y are different
            yf = pylab.fft2(y)
            y_flat = y.flatten()
            yy = pylab.dot(y_flat.transpose(), y_flat)
        else:
            # auto-correlation of x, avoid repeating a few operations
            yf = xf
            yy = xx
        # cross-correlation term in Fourier domain
        xyf = pylab.multiply(xf, pylab.conj(yf))
        # to spatial domain
        xyf_ifft = pylab.ifft2(xyf)
        #xy_complex = circshift(xyf_ifft, floor(x.shape/2))
        row_shift, col_shift = pylab.floor(pylab.array(x.shape)/2).astype(int)
        xy_complex = pylab.roll(xyf_ifft, row_shift,   axis=0)
        xy_complex = pylab.roll(xy_complex, col_shift, axis=1)
        xy = pylab.real(xy_complex)
        # calculate gaussian response for all positions
        scaling = -1 / (sigma**2)
        xx_yy = xx + yy
        xx_yy_2xy = xx_yy - 2 * xy
        k = pylab.exp(scaling * pylab.maximum(0, xx_yy_2xy / x.size))
        #print("dense_gauss_kernel x.shape ==", x.shape)
        #print("dense_gauss_kernel k.shape ==", k.shape)
        return k