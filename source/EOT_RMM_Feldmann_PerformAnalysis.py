#!/usr/bin/python
# -*- coding: utf-8 -*-

# In each image frame, according to the Ground Truth file,
# Put the object in the center of 256x128 formatted sub-image.

import cv2
import numpy as np
import coordinates_convert as corcon
import ast
import pylab
import KCFtracker_Status_MotionVector   as KCF_ST_MV
import utility as uti
import cmath
from   numpy.lib    import scimath
import scipy.linalg as sl

class rmm_ff(object):
    def init(self, cx, cy, orient, w, h):
        #time instant between 2 frames, 2.5s
        delta_t = 2.5
        #F means prediction equation  x_k = F*x_k-1 + Q
        #  u denotes postion, v denotes velocity.
        #  [ ux ]   [1, 0, 1, 0][ux]
        #  [ uy ]   [0, 1, 0, 1][uy]
        #  [ vx ] = [0, 0, 1, 0][vx]       + Q
        #  [ vy ]_k [0, 0, 0, 1][vy]_k-1

        # self.F  = np.array([[1., 0, 10., 0],
        #                     [0, 1., 0, 10.],
        #                     [0, 0, 1., 0],
        #                     [0, 0, 0, 1.]]) # for simulated data

        self.F  = np.array([[1., 0, 1., 0],
                            [0, 1., 0, 1.],
                            [0, 0, 1., 0],
                            [0, 0, 0, 1.]]) # for radar_data

        #H means measurement equation y_k = H*x_k + W
        self.H = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])  # matrix maps kinematic state into position

        # process  noise  covariance  for kinematic state
        self.Q = np.array([[100., 0,  0, 0],
                           [0,  100., 0, 0],
                           [0,   0,  1., 0],
                           [0,   0,  0, 1.]])
        # setting prior, position and velocity
        hat_r0 = np.array([cx, cy, 3., 3.])
        # shape variable: orientation and semi-axes lengths
        hat_p0 = np.array([orient, w, h])

        #init P0 for covariance error
        C_r0 = np.array([[900., 0, 0, 0],
                         [0, 900., 0, 0],
                         [0, 0, 16.,  0],
                         [0, 0, 0,  16.]])

        #measure noise, yes
        #self.R =  np.array([[2000., 0], [0, 80.]]) # for simulated_data

        self.R = np.array([[2., 0], [0, 8.]])   # for radar_tracking


        # Parameters for Random Matrix
        self.alpha0 = 50.
        self.tau = 10.
        self.T = 10.
        self.const_z = 1. / 4

        #intialized status
        self.x0 = hat_r0
        self.X0 = self.get_random_matrix_state(hat_p0)
        self.P0 = C_r0

    def get_random_matrix_state(self, ellipse_extent):
        '''only ellipse parameterization alpha is implemented
           % by Shishan     Yang
        '''
        alpha = ellipse_extent[0];
        eigen_val = ellipse_extent[1:3];
        eigen_vec = np.array([[np.cos(alpha), -np.sin(alpha)],
                              [np.sin(alpha), np.cos(alpha)]])
        diag_eigen = np.array([[eigen_val[0] ** 2, 0],
                               [0, eigen_val[1] ** 2]])
        A = np.dot(np.dot(eigen_vec, diag_eigen), eigen_vec.T)
        A = (A + A.T) / 2.
        return A


    def get_random_matrix_ellipse(self, rmm_extent):
        '''
        :param rmm_extent:
        :return:
        % notations and formmulas are from "Ellipse fitting based approach for extended object tracking"
        % Equation (9): a paper from Tsinghua 军人
        从协方差矩阵中提取出椭圆的alpha偏角，长短轴
        '''
        rho = (rmm_extent[0, 0] - rmm_extent[1, 1]) / (2. * rmm_extent[0, 1] + np.spacing(1))
        phi = cmath.atan(-rho + scimath.sqrt(1 + rho ** 2.))
        rmm_rotation = np.array([[cmath.cos(phi), -cmath.sin(phi)],
                                 [cmath.sin(phi), cmath.cos(phi)]])
        rmm_l_sq = np.diag(np.dot(np.dot(rmm_rotation.conj().T, rmm_extent), rmm_rotation))
        # for negative numbers, get complex number
        rmm_l = scimath.sqrt(rmm_l_sq)
        return (rmm_rotation, rmm_l, phi)

    def predictRMM(self, x, X, P, alpha, F, Q, T, tau):
        x_pred = np.dot(F, x)
        P_pred = np.dot(np.dot(F, P), F.conj().T) + Q

        X_pred = X
        alpha_pred = 2. + np.exp(-T / tau) * (alpha - 2.)
        return (x_pred, X_pred, P_pred, alpha_pred)

    def updateRMM(self, x, X, P, alpha, y, y_cov, R, n_k, H, const_z):
        '''
        Iterated update Random Matrix
        '''
        X_sqrt = sl.sqrtm(X)
        Y = const_z * X + R
        Y_sqrt_inv = sl.inv(sl.sqrtm(Y))
        S = np.dot(np.dot(H, P), H.T) + Y / (n_k*1.)
        S_sqrt = sl.sqrtm(S)
        # the bold_dot is: inverse_sqrt_S*(y-H*x)
        # bold_dot =  S_sqrt\(y - H * x);
        bold_dot = sl.solve(S_sqrt, (y - np.dot(H, x)).reshape(2, 1))
        tp = np.dot(X_sqrt, bold_dot)
        N_hat = np.dot(tp, tp.conj().T)

        K = np.dot(np.dot(P, H.T), sl.inv(S))

        alpha_update = alpha + n_k
        P_update = P - np.dot(np.dot(K, S), K.conj().T)
        x_update = x + np.dot(K, y - np.dot(H, x))
        X_update = (1. / alpha_update) * (alpha * X + N_hat + np.dot(np.dot(np.dot(X_sqrt, Y_sqrt_inv), y_cov), np.dot(X_sqrt, Y_sqrt_inv).conj().T))
        # # print 'x=', x, ';'
        # # print 'H=', H, ';'
        # print 'X=',X , ';'
        # # print 'P=', P, ';'
        # # print 'alpha=',alpha, ';'
        # # print 'y=',y, ';'
        # # print 'y_cov=', y_cov, ';'
        # # print 'R=',R, ';'
        # # print 'n_k = %f; const_z=%f;' % (n_k, const_z)
        # #
        # # print 'x_update=', x_update
        # print 'X_update=', X_update
        # # print 'P_update=', P_update
        # # print 'alpha_update=',alpha_update
        # print 'alpha*X=', (1 / alpha_update) *alpha*X
        # print 'N_hat='  , (1 / alpha_update) *N_hat
        # print 'star=',    (1 / alpha_update) *np.dot(np.dot(np.dot(X_sqrt, Y_sqrt_inv), y_cov), np.dot(X_sqrt, Y_sqrt_inv).conj().T)
        return (x_update, X_update, P_update, alpha_update)

    def plot_extent(self, ellipse, line_style, color, line_width):
        '''
        :param ellipse: 1x5, parameterization of one ellispe [m1 m2 alpha l1 l2]
        :param line_style:  definedthe same as in Matlab plot function
        :param color:       definedthe same as in Matlab plot function
        :param line_width:  definedthe same as in Matlab plot function
        :return:
        '''
        center = ellipse[0:2] / 1000.;
        theta = ellipse[2];
        l = ellipse[3:5] / 1000.;
        R = np.array(
            [[cmath.cos(theta), -cmath.sin(theta)], [cmath.sin(theta), cmath.cos(theta)]])  # rotation matrix

        alpha = np.arange(0, 2. * np.pi, np.pi / 100.)
        xunit = l[0] * np.cos(alpha)
        yunit = l[1] * np.sin(alpha)

        rotated = np.dot(R, np.vstack((xunit, yunit)))
        xpoints = rotated[0, :] + center[0]
        ypoints = rotated[1, :] + center[1]
        pylab.plot(xpoints, ypoints, linestyle=line_style, color=color, linewidth=line_width)



    def update(self, x_, X_, P_, alpha_, meas_mean, meas_spread, N):

        # predict　RMM
        (x_pr, X_pr, P_pr, alpha_pr) = self.predictRMM(x_, X_, P_, alpha_, self.F, self.Q, self.T, self.tau)


        x, X, P, alpha = self.updateRMM(x_pr, X_pr, P_pr, alpha_pr, meas_mean, meas_spread, self.R, N, self.H, self.const_z)

        # _, len_RMM, ang_RMM = self.get_random_matrix_ellipse(X)
        # print 'len_RMM = ', len_RMM
        # print 'ang_RMM = ', ang_RMM
        # print 'ang_RMM in Degree %f', ang_RMM * 180. / np.pi
        #
        # rmm_par = np.hstack((np.dot(self.H, x), ang_RMM, np.real(len_RMM)))
        # self.plot_extent(rmm_par, '-', 'g', 1)
        # pylab.show(block=False)

        # self.hat_x_RMM = hat_x_RMM
        # self.hat_X_RMM = hat_X_RMM
        # self.Cx_RMM = Cx_RMM
        # self.alpha  = alpha
        # print 'hat_x_RMM ', hat_x_RMM
        # print 'hat_X_RMM ', hat_X_RMM
        # print 'Cx_RMM'    , Cx_RMM
        # print 'alpha'     , alpha
        if alpha <= 2:
            print(' Error! alpha<2')
        return (x,X,P,alpha)
