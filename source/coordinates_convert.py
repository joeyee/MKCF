import cv2
import math
import time
import numpy as np
from   numba import njit
'''
%\Convert the radar range_azi polar matrix(g_radar_mat) into a displaying Descartes coordinate (cartesian)
(disp_mat)
'''


def polar2disp_njit(polarmat, dispmat):
    row, col = np.shape(polarmat)[0:2]
    if np.size(dispmat) == 0:
        dispmat = np.zeros((row + 1, row * 2))

    theta1row = np.linspace(0 - np.pi / 2, -np.pi / 2 + 2 * np.pi * col / 4096, col)
    theta = np.tile(theta1row, (row, 1))
    dist1row = np.arange(row)
    r = np.tile(dist1row, (col, 1)).T
    x = r * np.sin(theta)  + row
    y = -r * np.cos(theta) + row
    dispmat = polar2disp_core_numbajit(polarmat, dispmat, x, y, col, row)
    return dispmat


from numpy import arange
@njit
def polar2disp_core_numbajit(polarmat, dispmat, x, y, col, row):
    for j in arange(col):
      for i in arange(row):
           corx = int(x[i][j])
           cory = int(y[i][j])
           dispmat[cory][corx] = polarmat[i][j]
    return dispmat


#Pixel accessing method in normal python style
def polar2disp(polarmat, dispmat):
    # tstart = time.clock()
    row, col = np.shape(polarmat)[0:2]
    if np.size(dispmat)== 0:
        dispmat = np.zeros((row+1, row*2))
    # cv2.imshow('disp',polarmat)
    # cv2.waitKey()
    # for r in np.arange(row):
    #      for theta in np.arange(col):
    #           x = np.floor(r*np.sin(theta*2*np.pi/4096) + 2048)
    #           y = np.floor(-r.np.cos(theta*2*np.pi/4096) + 2048)
    #           print (x,y, r, theta)
    #           dispmat[x.astype('int')][y.astype('int')] = polarmat[r][theta]
    theta1row= np.linspace(0-np.pi/2, -np.pi/2+ 2*np.pi*col/4096, col)
    theta = np.tile(theta1row, (row,1))
    dist1row = np.arange(row)
    r = np.tile(dist1row , (col,1)).T
    x = r*np.sin(theta)  + row
    y = -r*np.cos(theta) + row
    # tend = (time.clock() - tstart) * 1000
    # print('cossin cost %f ms\n' % tend)
    tstart = time.clock()
    for j in np.arange(col):
      for i  in np.arange(row):
           corx = int(x[i][j])
           cory = int(y[i][j])
           dispmat[cory][corx] = polarmat[i][j]
    # tend  = (time.clock() - tstart)*1000
    # print('polar2disp cost %f s\n' % tend)
    return dispmat

