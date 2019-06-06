#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of KCF tracker, Added status monitor

import cv2
import numpy as np
#import polyROISelector
#import polyROI
#import coordinates_convert as corcon
#import ast
import pylab
pylab.ioff()


debug = False

class KCFTracker_status(object):
    """
    define the tracker model, inclding init(), update() submodels
    """
    # def __init__(self):
    #     return

    #init function is different for varied trackers
    def init(self, img, rect):
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

        self.rect = rect #rectangle contains the bounding box of the target
        #pos is the center postion of the tracking object (cy,cx)
        self.pos = pylab.array([rect[1] + rect[3]/2, rect[0] + rect[2]/2])
        self.posOffset = np.array([0,0],np.int)
        self.tlx = rect[0]
        self.tly = rect[1]
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
        x = self.get_subwindow(img, self.pos, window_sz, self.cos_window)
        # Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
        k = self.dense_gauss_kernel(self.sigma, x)
        #storing computed alphaf and z for next frame iteration
        self.alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))  # Eq. 7
        self.z = x

        #monitoring the tracker's self status, based on the continuity of psr
        self.self_status = 0
        #monitoring the collaborative status, based on the distance to the voted object bouding box center,  and on psr also.
        self.collaborate_status = 5

        self.collabor_container = np.ones((10,1),np.int)
        self.highpsr_container  = np.ones((10,1),np.int)
        self.FourRecentRects    = np.zeros((4,4),np.float)
        #return initialization status
        return True

    def  update(self, new_img):
        '''
        :param new_img: new frame should be normalized, for tracker_status estimating the rect_snr
        :return:
        '''
        self.canvas   = new_img.copy()
        self.trackNo +=1

        # get subwindow at current estimated target position, to train classifier
        x = self.get_subwindow(new_img, self.pos, self.window_sz, self.cos_window)
        # calculate response of the classifier at all locations
        k = self.dense_gauss_kernel(self.sigma, x, self.z)
        kf = pylab.fft2(k)
        alphaf_kf = pylab.multiply(self.alphaf, kf)
        response  = pylab.real(pylab.ifft2(alphaf_kf))     # Eq. 9
        self.response = response
        self.responsePeak = np.max(response)
        # target location is at the maximum response
        row, col = pylab.unravel_index(response.argmax(), response.shape)
        #roi rect's topleft point add [row, col]
        self.tly, self.tlx = self.pos - pylab.floor(self.window_sz / 2)

        #here the pos is not given to self.pos at once, we need to check the psr first.
        #if it above the threashhold(default is 5), self.pos = pos.
        pos = np.array([self.tly, self.tlx]) + np.array([row, col])

        #Noting, for pos(cy,cx)! for cv2.rect rect(x,y,w,h)!
        rect = pylab.array([pos[1]- self.target_sz[1]/2, pos[0] - self.target_sz[0]/2, self.target_sz[1], self.target_sz[0]])
        rect = rect.astype(np.int)
        self.rect = rect
        self.psr, self.trkStatus = self.tracker_status(col, row, response, rect, new_img)
        self.pos  = pos

        # #bad quality tracking results
        # if self.psr <= 5  and self.trackNo >=5:
        #     # computing offset based on the last 4 frame's obj_bbox'center.
        #     # using the average center shift as the (offset_x, offset_y)
        #     dif_rect = []
        #     #for iter in [-1, -2, -3]:
        #     for iter in [-1,-2,-3 ]:
        #         dif_rect.append(np.array(self.FourRecentRects[iter]) - np.array(self.FourRecentRects[iter - 1]))
        #     offset_rect = np.mean(dif_rect, 0)
        #     offset = (offset_rect[0] + offset_rect[2] / 2, offset_rect[1] + offset_rect[3] / 2)
        #     print('Tracker offset is activited (%d, %d)' % (offset[0], offset[1]))
        #     self.pos = self.pos + np.array([ offset[1], offset[0] ])
        #     # rect = pylab.array([self.pos[1] - self.target_sz[1] / 2, self.pos[0] - self.target_sz[0] / 2, self.target_sz[1], self.target_sz[0]])
        #     # rect = rect.astype(np.int)
        #     # self.FourRecentRects[self.trackNo % 4] = rect
        # else:
        #     self.pos = pos
        #     self.FourRecentRects[self.trackNo % 4] = rect

        #if self.psr <= 5:
        #     # computing offset based on the last 4 frame's obj_bbox'center.
        #     # using the average center shift as the (offset_x, offset_y)
        #
        #     self.pos = self.pos + self.posOffset
        #     print self
        #     print('Tracker Default Offset is activited (%d, %d)' % (self.posOffset[1], self.posOffset[0]))

        #
        #     # rect = pylab.array([self.pos[1] - self.target_sz[1] / 2, self.pos[0] - self.target_sz[0] / 2, self.target_sz[1], self.target_sz[0]])
        #     # rect = rect.astype(np.int)
        #     # self.FourRecentRects[self.trackNo % 4] = rect
        #else:
        #     self.pos = pos
        #     self.FourRecentRects[self.trackNo % 4] = rect
        #     if self.trackNo >= 5:
        #         dif_rect = []
        #         # for iter in [-1, -2, -3]:
        #         for iter in [-1, -2, -3]:
        #             dif_rect.append(np.array(self.FourRecentRects[iter]) - np.array(self.FourRecentRects[iter - 1]))
        #         offset_rect = np.mean(dif_rect, 0)
        #         offset = (offset_rect[0] + offset_rect[2] / 2, offset_rect[1] + offset_rect[3] / 2)
        #         self.posOffset =  np.array([offset[1], offset[0]])


        #print ('tracker\'status:res_win_ave,max,psr, rect_snr', self.trkStatus)
        # if debug == True:
        #     if self.trackNo == 1:
        #         #pylab.ion()  # interactive mode on
        #         self.fig, self.axes = pylab.subplots(ncols=3)
        #         self.fig.show()
        #         # We need to draw the canvas before we start animating...
        #         self.fig.canvas.draw()
        #
        #         k_img = self.axes[0].imshow(k,animated=True)
        #         x_img = self.axes[1].imshow(x,animated=True)
        #         r_img = self.axes[2].imshow(response,animated=True)
        #
        #         self.subimgs = [k_img, x_img, r_img]
        #         # Let's capture the background of the figure
        #         self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
        #
        #         pylab.show(block=False)
        #     else:
        #         self.subimgs[0].set_data(k)
        #         self.subimgs[1].set_data(x)
        #         self.subimgs[2].set_data(response)
        #         items = enumerate(zip(self.subimgs, self.axes, self.backgrounds), start=1)
        #         for j, (subimg, ax, background) in items:
        #             self.fig.canvas.restore_region(background)
        #             ax.draw_artist(subimg)
        #             self.fig.canvas.blit(ax.bbox)
        #         pylab.show(block=False)

        #only update when tracker_status's psr is high
        if (self.psr > 10):
            #computing new_alphaf and observed x as z
            x = self.get_subwindow(new_img, self.pos, self.window_sz, self.cos_window)
            # Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
            k = self.dense_gauss_kernel(self.sigma, x)
            new_alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))  # Eq. 7
            new_z = x

            # subsequent frames, interpolate model
            f = self.interpolation_factor
            self.alphaf = (1 - f) * self.alphaf + f * new_alphaf
            self.z = (1 - f) * self.z + f * new_z
        ok = 1
        return ok, rect, self.psr, response

    def status_monitor(self, psr, dist, psr_threash = 10, dist_theash = 100):

        #monitoring the individual tracker's status
        if psr >= psr_threash:
            self.self_status += 1
            self.highpsr_container[self.trackNo % 10] = 1
        else:
            self.self_status -= 1
            self.highpsr_container[self.trackNo % 10] = 0

        #monitoring the tracker's contribution on the multi trackers' voted object bouding box.
        if psr >= psr_threash/2 and dist <= dist_theash:
            self.collaborate_status += 1
            self.collabor_container[self.trackNo % 10] = 1
        # if psr < psr_threash/2 and dist > dist_theash:
        #     self.collaborate_status -= 1
        #     self.collabor_container[ self.trackNo%10 ] = 0
        if dist > dist_theash:
            self.collaborate_status -= 1
            self.collabor_container[ self.trackNo%10 ] = 0

    #def refresh_position(self, offset):
    def refresh_position(self, objRect):

        self.pos = np.array([objRect[1], objRect[0]])
        #self.posOffset = np.array([offset[1], offset[0]])
        ##offset is (dx,dy), pos is (cy,cx)
        #self.pos = self.pos + self.posOffset

        print ('kcf relocation by mkcf')
        # rect = pylab.array([self.pos[1] - self.target_sz[1] / 2, self.pos[0] - self.target_sz[0] / 2, self.target_sz[1], self.target_sz[0]])
        # rect = rect.astype(np.int)
        # self.FourRecentRects[self.trackNo % 4] = rect

        # self.posOffset = np.array([offset[1], offset[0]])

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

        # if debug:
        #     print("Out max/min value==", out.max(), "/", out.min())
        #     pylab.figure()
        #     pylab.imshow(out, cmap=pylab.cm.gray)
        #     pylab.title("cropped subwindow")

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

    def response_win_ave_max(self, response, cx, cy, winsize):
        '''
        computing the average and maximum value in a monitor window of response map
        :param response:
        :param cx:
        :param cy:
        :return:
        '''
        # res_monitor_windows_size
        tlx = int(max(0, cx - winsize / 2))
        tly = int(max(0, cy - winsize / 2))
        brx = int(min(cx + winsize / 2, response.shape[1]))
        bry = int(min(cy + winsize / 2, response.shape[0]))

        reswin = response[tly:bry, tlx:brx]
        # average value in res_monitor_windows
        res_win_ave = np.mean(reswin)
        res_win_max = np.max(reswin)

        sidelob = response.copy()
        exclude_nums = (brx-tlx)*(bry-tly)
        #excluding the peak_neighbour
        sidelob[tly:bry, tlx:brx] = 0
        sidelob_mean = np.sum(sidelob)/(sidelob.shape[0]*sidelob.shape[1] - exclude_nums)

        sidelob_var  = (sidelob - sidelob_mean)**2
        #exclude the peak_neighbour
        sidelob_var[tly:bry, tlx:brx] = 0
        sidelob_var = np.sum(sidelob_var)/(sidelob.shape[0]*sidelob.shape[1] - exclude_nums)

        #peak to sidelobe ratio
        psr = (res_win_max-sidelob_mean)/np.sqrt(sidelob_var+np.spacing(1))

        return res_win_ave, res_win_max, psr

    def kmatrix_blob_ratio(self, kmatrix, blob):
        '''
        computing the energy ratio in blob/in kmatrix
        :param kmatrix:
        :param blob:
        :return:
        '''
        tlx = self.tlx
        tly = self.tly
        poly = blob['Polygon']
        if len(poly)==0:
            print ('no blob in the kmatrix')
        offset = np.tile([[[tlx, tly]]], (len(poly), 1, 1))

        poly_in_kmatrix = poly - offset
        img = np.zeros(kmatrix.shape, np.uint)
        img = cv2.fillConvexPoly(img, poly_in_kmatrix, [1], 8)

        ratio = np.sum(kmatrix*img)/np.sum(kmatrix)

        cv2.imshow('blob in kwindow', img)
        pylab.figure()
        pylab.imshow(kmatrix)
        pylab.figure()
        pylab.imshow(img)
        pylab.show(block=False)

    def tbb_blob_ratio(self, blob):
        '''
        computing the overlapped ratio of tracker's rect and blob's boundingbox
        :param blob:
        :return:
        '''
        r1 = (self.pos[1]- self.target_sz[1]/2, self.pos[0] - self.target_sz[0]/2, self.target_sz[1], self.target_sz[0])
        r2 = blob['BoundingBox']
        intx = min(r1[0] + r1[2], r2[0] + r2[2]) - max(r1[0], r2[0])
        inty = min(r1[1] + r1[3], r2[1] + r2[3]) - max(r1[1], r2[1])
        if intx < 0 or inty < 0:
            intersection = 0.
        else:
            intersection = intx * inty
        inter_ratio = intersection / (r1[2] * r1[3] + r2[2] * r2[3] - intersection + np.spacing(1))
        return inter_ratio



    def tracker_status(self, cx, cy, response, rect, frame):
        '''
        #Monitoring the tracker's status
        :param cy: max valuse's row in response
        :param cx: max value's col in response
        :param rect: estimated bounding box
        :param response: response map
        :param frame: current frame
        :return:
        '''
        #response_windows_status
        #rect with less similarity object, will has little res_win_ave,max,and psr.
        #for winsize in [4,8,16,32]:
        winsize = 12
        res_win_ave, res_win_max, psr = self.response_win_ave_max(response, cx, cy, winsize)
        res_win_status=(res_win_ave, res_win_max, psr)

        # pylab.figure()
        # pylab.imshow(response)
        # pylab.show(block=False)

        #rect_snr monitoring the intensity in the rect/
        #rect with no echos has little rect_snr.
        roi = self.get_imageROI(frame, rect)
        rect_snr = np.sum(roi)/(roi.shape[0]*roi.shape[1])

        #km_blob_energy_ratio = self.kmatrix_blob_ratio(kmatrix, blob)
        #tb_blob_inter_ratio  = self.tbb_blob_ratio(blob)

        trk_status = [res_win_status, rect_snr]

        return psr, trk_status

    def update_ret_response(self, new_img):
        '''
        :param new_img: new frame should be normalized, for tracker_status estimating the rect_snr
        :return:
        '''
        self.canvas = new_img.copy()
        self.trackNo += 1

        # get subwindow at current estimated target position, to train classifier
        x = self.get_subwindow(new_img, self.pos, self.window_sz, self.cos_window)
        # calculate response of the classifier at all locations
        k = self.dense_gauss_kernel(self.sigma, x, self.z)
        kf = pylab.fft2(k)
        alphaf_kf = pylab.multiply(self.alphaf, kf)
        response = pylab.real(pylab.ifft2(alphaf_kf))  # Eq. 9

        # target location is at the maximum response
        row, col = pylab.unravel_index(response.argmax(), response.shape)
        # roi rect's topleft point add [row, col]
        self.tly, self.tlx = self.pos - pylab.floor(self.window_sz / 2)

        # here the pos is not given to self.pos at once, we need to check the psr first.
        # if it above the threashhold(default is 5), self.pos = pos.
        pos = np.array([self.tly, self.tlx]) + np.array([row, col])

        # Noting, for pos(cy,cx)! for cv2.rect rect(x,y,w,h)!
        rect = pylab.array(
            [pos[1] - self.target_sz[1] / 2, pos[0] - self.target_sz[0] / 2, self.target_sz[1], self.target_sz[0]])
        rect = rect.astype(np.int)

        self.psr, self.trkStatus = self.tracker_status(col, row, response, rect, new_img)
        self.pos = pos
        #only update when tracker_status's psr is high
        if (self.psr > 10):
            #computing new_alphaf and observed x as z
            x = self.get_subwindow(new_img, self.pos, self.window_sz, self.cos_window)
            # Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
            k = self.dense_gauss_kernel(self.sigma, x)
            new_alphaf = pylab.divide(self.yf, (pylab.fft2(k) + self.lambda_value))  # Eq. 7
            new_z = x

            # subsequent frames, interpolate model
            f = self.interpolation_factor
            self.alphaf = (1 - f) * self.alphaf + f * new_alphaf
            self.z = (1 - f) * self.z + f * new_z
        ok = 1
        return ok, rect, self.psr, response
    #TODO blob_seg's initial position should be given by last frame's estimated position.(done)
    #TODO score_matrix 的列可以作为voted_blob 对 Tracker的打分(done)