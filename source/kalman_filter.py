#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.linalg as sl

class kalman_filter(object):
    def init(self, cx, cy):
        self.X0 = np.array([[cx],
                            [0],
                            [cy],
                            [0]])

        self.P0 = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])

        self.phi = np.array([[1, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

        self.arh = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0]])
        varx = 4
        vary = 6
        self.Q_n  = np.array([[0,0,0,0],
                              [0,varx, 0, 0],
                              [0,0,0,0],
                              [0, 0, 0, vary]])
        varmx = 2
        varmy = 3
        self.Q_m  = np.array([[varmx, 0],
                             [0, varmy]])
    def update(self, X_, P_, cx, cy):
        '''
        :param X_: last status
        :param P_: last error variance
        :param cx: measured centerx
        :param cy: measured centery
        :return: estimated currenet status X, and error variance
        '''
        Z   = np.array([[cx],
                        [cy]])
        Q_n = self.Q_n
        Q_m = self.Q_m
        phi = self.phi
        arh = self.arh

        X_pr = np.dot(phi, X_)
        P_pr = np.dot(np.dot(phi, P_), P_.T) + Q_n
        Y_pr = Z - np.dot(arh, X_pr)
        S_pr = np.dot(np.dot(arh, P_pr), arh.T) + Q_m
        S_pr_inv = sl.inv(S_pr)
        K    = np.dot(np.dot(P_pr, arh.T), S_pr_inv)
        X    = X_pr + np.dot(K, Y_pr)
        P    = P_pr - np.dot(np.dot(K, arh), P_pr)
        return (X, P, X_pr)



def kalmanFilter(phi, arh, X_, P_, Z, Q_n, Q_m):
    '''

    :param phi: status translation matrix
    :param X_:  status vector in previous time
    :param P_:  error coviarance in previous time
    :param Z:   current observation
    :return: X, P
    '''

    X_pr = np.dot(phi, X_)
    P_pr = np.dot(np.dot(phi, P_), P_.T) + Q_n
    Y_pr = Z - np.dot(arh, X_pr)
    S_pr = np.dot(np.dot(arh, P_pr), arh.T) + Q_m
    S_pr_inv = sl.inv(S_pr)
    K    = np.dot(np.dot(P_pr, arh.T), S_pr_inv)
    X    = X_pr + np.dot(K, Y_pr)
    P    = P_pr - np.dot(np.dot(K, arh), P_pr)
    return (X, P)

if __name__=='__main__':
    phi = np.array([[1, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1]])

    arh = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0]])
    varx = 4
    vary = 6
    Q_n  = np.array([0,0,0,0],
                    [0,varx, 0, 0],
                    [0,0,0,0],
                    [0, 0, 0, vary])
    varmx = 2
    varmy = 3
    Q_m  = np.array([varmx, 0],
                    [0, varmy])

    x0 = 12
    y0 = 8
    X0 = np.array([[x0],
                   [1],
                   [y0]
                   [2]])

    P0 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]])


    Zs = []
    X = x0
    for t in range(100):
        Z = np.dot(phi, X)
        X = Z
        Zs.append(Z)

    #run kalman filter for estimation
    X_ = X0
    P_ = P0
    for Z in Zs:
        (X,P) = kalmanFilter(phi, arh, X_, P_, Z, Q_n, Q_m)
        X_ = X
        P_ = P