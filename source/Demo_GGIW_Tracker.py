"""
Implement the GGIW-single ETT object tracking on Marine-radar.
Author: Yi Zhou
Date:   20190216
Contact: yi.zhou@dlmu.edu.cn
Refer to:
(1)Fusion Group of Geottingen university ETT toolbox on Github:
https://github.com/Fusion-Goettingen/ExtendedTargetTrackingToolbox/
(2)Karl_ICIF16 GGIW paper:
“Gamma Gaussian inverse-Wishart Poisson multi-Bernoulli Filter for Extended Target Tracking ” ICIF16 Karl
"""
import cv2
import numpy    as np
import utility  as uti
import numpy.linalg as la
from   numpy.lib    import scimath
import sys
import time
import cmath
#add ETT_TOOLBOX to the working directory for 'from models import spline'
sys.path.insert(0,'/Users/yizhou/code/MKCF_MAC_Python3.6/source/ETT_TOOLBOX/')
from ETT_TOOLBOX.models import GgiwTracker


def load_zhicheng_GT():
    '''
    Set the path and GT files name, according start_id, end_id.
    :return:
    '''
    FilePath = '../ground_truth/'
    GtParam = {'ALice':{'FileName':'zhicheng20130808_22.xx_60-400_polarGT_Alice.txt',   'StartId':60, 'EndId':400},
              'Billy':{'FileName':'zhicheng20130808_22.xx_2-215_polarGT_Billy.txt',     'StartId':2,  'EndId':226},
              #'Camen':{'FileName':'zhicheng20130808_22.xx_2-408_polarGT_Camen.txt',     'StartId':2,  'EndId':408},
              'Camen': {'FileName': 'zhicheng20130808_22.xx_2-408_polarGT_Camen.txt', 'StartId': 75, 'EndId': 335},
              'Dolphin':{'FileName':'zhicheng20130808_22.xx_72-316_polarGT_Dolphin.txt','StartId':72+1, 'EndId':316},
              'Ellen':{'FileName':'zhicheng20130808_22.xx_129-408_polarGT_Ellen.txt',   'StartId':129,'EndId':408}
              }
    #将GtParam里面的文件路径参数赋予load_GT函数，分析出Gt的每帧位置信息，存入GtData
    GtInfo = {}
    for key in GtParam:
        GtElem      = GtParam[key]
        fname       = FilePath + GtElem['FileName']
        GtData      = uti.get_obj_dict(fname)
        start_id    = GtElem['StartId']
        end_id      = GtElem['EndId']
        GtInfo[key] = {'GtData':GtData, 'StartId':start_id, 'EndId':end_id}
    return GtInfo

def get_covariance_matrix_ellipse(cov):
    '''
    :param rmm_extent:
    :return:
    % notations and formmulas are from "Ellipse fitting based approach for extended object tracking"
    % Equation (9): a paper from Tsinghua 军人
    从协方差矩阵中提取出椭圆的alpha偏角，长短轴
    '''
    rho = (cov[0, 0] - cov[1, 1]) / (2. * cov[0, 1] + np.spacing(1))
    phi = cmath.atan(-rho + scimath.sqrt(1 + rho ** 2.))
    rmm_rotation = np.array([[cmath.cos(phi), -cmath.sin(phi)],
                             [cmath.sin(phi), cmath.cos(phi)]])
    rmm_l_sq = np.diag(np.dot(np.dot(rmm_rotation.conj().T, cov), rmm_rotation))
    # for negative numbers, get complex number
    rmm_l = scimath.sqrt(rmm_l_sq)
    return (rmm_rotation, rmm_l, phi)

def  monitor_weight(canvas_frame, gtTbb,  obj_rect, offset, tbbs, psrs, sigma_factor= 3./4):
    # position array
    posxs = tbbs[:, 0] + tbbs[:, 2] / 2
    posys = tbbs[:, 1] + tbbs[:, 3] / 2

    gtposx = gtTbb[0] + gtTbb[2]/2
    gtposy = gtTbb[1] + gtTbb[3]/2

    dist2gt = np.sqrt((posxs - gtposx)**2 + (posys - gtposy)**2)


    cx, cy = (obj_rect[0] + obj_rect[2] / 2, obj_rect[1] + obj_rect[3] / 2)
    w = obj_rect[2] * 2
    h = obj_rect[3] * 2

    sigma_w = sigma_factor * w
    sigma_h = sigma_factor * h

    ofx, ofy = offset[0:2]
    distxs = np.float32(posxs) - (cx + ofx)
    distys = np.float32(posys) - (cy + ofy)

    # priority probability
    pprob = pylab.exp(-0.5 * ((distxs / sigma_w) ** 2 + (distys / sigma_h) ** 2))

    norm_psrs = psrs / np.sum(psrs)
    # psrs = psrs.reshape((len(psrs), 1))
    # psrs = np.tile(psrs, (1, 4))

    # NOTE, here the pprob are normalized before using!!! so it's not gaussian distribution anymore.
    norm_pprob = pprob / np.sum(pprob)
    #weights = norm_psrs * norm_pprob
    weights    = psrs * pprob

    weights     = weights / np.sum(weights)
    weights_mat = weights.reshape(len(weights), 1)
    weights_mat = np.tile(weights_mat, (1, 4))
    # pbs = pbs.reshape(len(pbs),1)
    # pbs = np.tile(pbs,(1,4))
    # obj_bbox is the average rect of all the tbbs with responding weights, given by tracker's psr and prio-probability
    obj_bbox_array = np.int0(np.sum(tbbs * weights_mat, axis=0))
    obj_bbox = (obj_bbox_array[0], obj_bbox_array[1], obj_bbox_array[2], obj_bbox_array[3])

    #sort the struct array to check the correlation between dist, pp, and weights.
    stype = [('dist', float), ('prioProb', float), ('normPP', float), ('psr', float), ('normPsr', float), ('weight', float), ('x', int), ('y', int), ('w',int), ('h', int)]
    values = []
    for di, pp, npp, psr, npsr, wt, tbb in zip(dist2gt, pprob, norm_pprob, psrs, norm_psrs, weights, tbbs):
        values.append((di, pp, npp, psr, npsr, wt, tbb[0], tbb[1], tbb[2], tbb[3]))
    valueArray      = np.array(values, dtype = stype)

    sortValuesArray_dist   = np.sort(valueArray, order='dist')
    sortValuesArray_weights= np.sort(valueArray, order='weight')

    nrows = sortValuesArray_dist.shape[0]
    ncount = 10
    for i in range(nrows):
        if i < ncount:
            sarray = sortValuesArray_dist[i]
            rect   = (sarray['x'],sarray['y'], sarray['w'], sarray['h'])
            uti.draw_rect(canvas_frame, rect )
            print( '+dist2Gt %3.2f, pp %0.4f, normpp %0.4f, psr %2d, normPsr %0.4f, weight %0.4f' % \
                (sarray['dist'], sarray['prioProb'], sarray['normPP'], sarray['psr'], sarray['normPsr'], sarray['weight']))

    #Draw the top-pest weights:
    for i in range(-1,-ncount, -1):
            sarray = sortValuesArray_weights[i]
            rect   = (sarray['x'],sarray['y'], sarray['w'], sarray['h'])
            uti.draw_rect(canvas_frame, rect, (0,0,255) )
            print( '-dist2Gt %3.2f, pp %0.4f, normpp %0.4f, psr %2d, normPsr %0.4f, weight %0.4f' % \
                (sarray['dist'], sarray['prioProb'], sarray['normPP'], sarray['psr'], sarray['normPsr'], sarray['weight']))

    cv2.imshow('test', canvas_frame)
    cv2.waitKey(1)

    return obj_bbox


def weight_tbox(obj_rect, offset, tbbs, psrs, sigma_factor = 1./4):
    '''
    center means the gaussian prior's maximum position, offset should be add before this call
    sigma_w, sigma_h 2d gaussian has two sigmas for row and col
    tbbs, tracked_bounding_box'list to np.array.
    psrs, tbbs' psr score list to np.array.
    :param center:
    :param sigma_w: shoulb be set by blob_seg's roi windows' shape, roi_w*sigma_factor
    :param sigma_h:
    :param tbbs:
    :param psrs:
    :return: maximum posterior probability box
    '''
    if tbbs.shape[0] == 0 or psrs.shape[0] == 0:
        print( 'Error: empty tracker_tbbs list or psr list in weight_tbox function.\n Return last frame\'s position \n')
        return obj_rect

    #position array
    posxs = tbbs[:,0] + tbbs[:,2]/2
    posys = tbbs[:,1] + tbbs[:,3]/2

    cx, cy = (obj_rect[0]+obj_rect[2]/2, obj_rect[1]+obj_rect[3]/2)
    w      = obj_rect[2] * 2
    h      = obj_rect[3] * 2

    sigma_w = sigma_factor*w
    sigma_h = sigma_factor*h

    ofx, ofy = offset[0:2]
    distxs = np.float32(posxs) -(cx + ofx)
    distys = np.float32(posys) -(cy + ofy)

    #priority probability
    pprob = pylab.exp(-0.5 * ((distxs / sigma_w) ** 2 + (distys / sigma_h) ** 2))

    psrs = psrs / (np.sum(psrs)+np.spacing(1))
    # psrs = psrs.reshape((len(psrs), 1))
    # psrs = np.tile(psrs, (1, 4))

    #NOTE, here the pprob are normalized before using!!! so it's not gaussian distribution anymore.
    norm_pprob = pprob / (np.sum(pprob) + np.spacing(1))
    weights = psrs * norm_pprob

    weights = weights / np.sum(weights)
    weights = norm_pprob.reshape(len(weights), 1)
    weights = np.tile(weights, (1, 4))
    # pbs = pbs.reshape(len(pbs),1)
    # pbs = np.tile(pbs,(1,4))
    # obj_bbox is the average rect of all the tbbs with responding weights, given by tracker's psr and prio-probability
    obj_bbox_array = np.int0(np.sum(tbbs * weights, axis=0))
    obj_bbox = (obj_bbox_array[0], obj_bbox_array[1],obj_bbox_array[2], obj_bbox_array[3])
    return obj_bbox

#For controlling the detailed display information.
DETAIL_MODE = False
def load_frame(obj_name, trial_times, GtDict, roi_mask, start_id, end_id, ParamOptions):
    """
    :param obj_name:     the name of the tracked object
    :param trial_times:  repeat time index for a specific target
    :param GtDict:       ground truth information for a target
    :param roi_mask:     mask off the bridge echos.
    :param start_id:     start frame id of the target
    :param end_id:       ending frame id of the target
    :param ParamOptions: parameter settings for the blog segmentations.
    :return:
    """
    print( 'Tracking %s:...' % obj_name)
    resPath = '../results/Res_EOTGGIW/'

    tracker_res_name  = resPath + obj_name + '_blobs_%02d_option.txt' % trial_times
    tracker_tbb_name  = resPath + obj_name + '_EOTGGIW_Tbbs.txt'
    track_res_file    = open(tracker_res_name, 'w')
    tracker_tbbs_file = open(tracker_tbb_name, 'w')

    segBlobParam     = ParamOptions['BlobSeg']
    VotePsrThreash   = ParamOptions['VotePsr']
    VoteDistThreash  = ParamOptions['VoteDistThreash']
    BlobScoreThreash = ParamOptions['BlobScore']
    PrioSigmaFactor  = ParamOptions['PrioSigmaFactor']
    NumberOfTrackers = ParamOptions['NumberOfTrackers']


    frame = np.array([])  # for storing image
    #tracker_dict    = {}
    tracker_list = []
    track_rect_list = []
    #ignore the damaged frames
    damaged_frames = [1, 116, 232, 348]
    frame_id = 0
    # starting frame_id
    #start_id = 60
    frame_counter     = 0
    total_time_consum = 0
    frame_mode = 'polar'

    # reading the sequential 4 files in a loop
    for i in np.arange(4):
        file_prefix = '/Users/yizhou/Radar_Datasets/Zhicheng/zhicheng_20130808_22.'
        file_suffix = '_5minutes_600x2048.double.data'
        data_name = '%s%d%s' % (file_prefix, 28 + i * 5, file_suffix)
        print( data_name)
        fdata_obj = open(data_name, 'rb')

        while True:
            #tstart = time.clock()
            if frame_id > end_id:
                #using empty frame to break the while loop
                frame = np.array([])
                break
            else:
                # read a frame from local file
                frame = np.fromfile(fdata_obj, 'float64', 600 * 2048)
            if frame.size != 0:
                frame_id += 1
                if frame_id >= start_id and frame_id not in damaged_frames:
                    key = 'frame %d' % frame_id
                    objGtElem = {}
                    if key not in GtDict:
                        print ('%s is not in GT files!!!' % key)
                    else:
                        objGtElem = GtDict[key]

                    frame = frame.reshape([2048, 600])
                    frame = frame.T
                    # 通过计算均值mean
                    # fmean = frame.mean(axis=1)
                    # mfm   = np.tile(fmean,(600,1)).transpose()
                    frame = frame - 4000  # frame.mean()
                    # 归一化操作
                    # dframe is for computing in double float
                    dframe = uti.frame_normalize(frame)
                    # #uframe is for displaying,optical flow computing and finding blob
                    uframe = (dframe * 255).astype(np.uint8)
                    canvas_polar = cv2.cvtColor(uframe, cv2.COLOR_GRAY2BGR)
                    cv2.putText(canvas_polar, obj_name+' frame ' + str(frame_id), org=(10, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                                thickness=2, lineType=cv2.LINE_AA)
                    #cv2.imshow('polar', canvas_polar)

                    # for cartesian-coordinates
                    # dispmat = corcon.polar2disp_njit(dframe, np.array([]))
                    # dispmat = uti.frame_normalize(dispmat)
                    # udispmat = (dispmat * 255).astype(np.uint8)
                    # canvas_disp = cv2.cvtColor(udispmat, cv2.COLOR_GRAY2BGR)
                    # cv2.putText(canvas_disp, 'frame ' + str(frame_id), org=(10, 50),
                    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                    #             thickness=2, lineType=cv2.LINE_AA)
                    # #cv2.imshow('disp', canvas_disp)
                    # # cv2.moveWindow('polar',0,0)
                    # cv2.waitKey(1)

                    if frame_mode == 'polar':
                        tframe  = dframe * roi_mask
                        # #echoThreash plays a role of CFAR.
                        # echoThreash = np.mean(tframe)
                        tuframe = uframe
                        canvas_frame = canvas_polar
                    # else:
                    #     tframe  = dispmat
                    #     tuframe = udispmat
                    #     canvas_frame = canvas_disp

                    if frame_id == start_id :
                        # init previous frame
                        preframe = tuframe
                        obj_bbox = objGtElem['BoundingBox']

                        cx = obj_bbox[0] + obj_bbox[2]/2
                        cy = obj_bbox[1] + obj_bbox[3]/2

                        # tracker config
                        dt = 2.5  # step time difference
                        sa_sq = 1.5 ** 2

                        config = {
                            'steps': end_id - start_id + 1,  # total time steps
                            'd': 2,  # dimensions
                            's': 2,  #
                            'sd': 4,  #
                            'lambda_d': 0.999,  # exp(-dt/tal)
                            'eta': 1.0 / 0.999,  # factor for alpha iterative decrease
                            'n_factor': 1.0,  #
                            'f': np.asarray([[1, dt], [0, 1]], dtype='f4'),  # sub-matrix of translation matrix
                            'q': sa_sq * np.outer(np.asarray([dt ** 2 / 2.0, dt]), np.asarray([dt ** 2 / 2.0, dt])),
                        # transition  noise
                            'h': np.asarray([[1, 0]], dtype='f4'),  # sub measure matrix
                            'init_m': [cx, cy, 4, 2],  # initial state mean (cx, cy, vx, vy)
                            'init_c':  np.outer(np.asarray([dt ** 2 / 2.0, dt]),
                                                                np.asarray([dt ** 2 / 2.0, dt])), #2 ** 2 * sa_sq *
                        # initial state sub-covariance
                            'init_alpha': 350,#5,  # Gammar distribution parameter alpha
                            'init_beta': 120,  # gammar distribution parameter beta
                            'init_v': np.diag([400, 200]),  # shape matrix in iw distribution
                            'init_nu': 350, #5,  # free degree in iw_distribution
                        }
                        # init ggiwETT tracker

                        tracker = GgiwTracker(dt=dt, **config)

                        blob_list, seg_roi, sub_blob_list = uti.blob_seg(tframe, obj_bbox)
                        obj_blob, blob_score, blob_id = uti.vote_blob(blob_list, [obj_bbox])
                        if obj_blob=={}:
                            print ('No blob is found in first frame!!!')
                        else:
                            blob_key = 'frame ' + str(frame_id)
                            linedict = {}
                            linedict[blob_key] = obj_blob
                            track_res_file.writelines(str(linedict) + '\n')
                            track_res_file.flush()

                        bb_linedict = {}
                        tbb_key = 'frame ' + str(frame_id)
                        bb_linedict[tbb_key] = {'BoundingBox':obj_bbox}
                        tracker_tbbs_file.writelines(str(bb_linedict) + '\n')

                        track_rect_list.append(obj_bbox)

                        uti.draw_rect(canvas_frame, obj_bbox, (0, 0, 255), 2)


                    if frame_id > start_id:
                        # only counting the tracking time
                        tstart = time.clock()
                        print('frame id %d' % frame_id)
                        #seg window is initialized by last frame's boundingbox
                        blob_list, seg_roi, sub_blob_list = uti.blob_seg(tframe, obj_bbox,
                                                                         segBlobParam['MorphIter'], segBlobParam['DilateIter'], segBlobParam['DistThreash'],
                                                                         segBlobParam['SubWindow'])

                        cx = obj_bbox[0] + obj_bbox[2] / 2
                        cy = obj_bbox[1] + obj_bbox[3] / 2
                        dist_min = 200
                        dist = 0
                        blob_id = 0
                        for bid, blob in enumerate(blob_list):
                            (bcx, bcy) = blob['Centroid']
                            dist = np.sqrt((cx - bcx) ** 2 + (cy - bcy) ** 2)
                            if dist < dist_min:
                                dist_min = dist
                                blob_id = bid
                        if dist_min < 200 and len(blob_list) > 0:
                            blob = blob_list[blob_id]

                            poly_blob = np.array([blob['Polygon']])
                            img = np.zeros((canvas_frame.shape[0], canvas_frame.shape[1]), np.uint8)
                            img = cv2.fillConvexPoly(img, poly_blob, [255], 8)

                            # cv2.imshow('img',img)
                            # cv2.waitKey()
                            (idy, idx) = np.nonzero(img == 255)

                            dt_z = np.dtype([('ts', 'i4'),
                                             ('xy', 'f8', (2,))])
                            measures = np.array([])
                            measures = np.zeros(len(idy), dtype=dt_z)
                            for cnt, (dx, dy) in enumerate(zip(idx, idy)):
                                measures['ts'][cnt] = frame_counter
                                measures['xy'][cnt] = np.array([dx, dy])

                            tracker.step(measures)
                            print('number of pixels in the target %d\n' % len(measures))

                            #compute the ellipse shape from the ggiw_posterior.
                            v_hat = tracker._ggiw_post['v'][tracker._uc, :, :]
                            v_hat /= np.maximum(1.0, tracker._ggiw_post['nu'][tracker._uc, None, None] - 2 * tracker._1_d_1)

                            w, v = la.eig(v_hat)
                            est_theta = np.arctan2(v[1, 0], v[0, 0])
                            rad_to_deg = 180.0 / np.pi
                            est_theta *= rad_to_deg

                            est_cx = tracker._ggiw_post['m'][tracker._uc, 0]
                            est_cy = tracker._ggiw_post['m'][tracker._uc, 1]

                            est_w  = w[0]
                            est_h  = w[1]

                            est_width, est_height = np.sqrt(la.eigvals(v_hat))
                            est_angle = 0.5 * np.arctan2(2 * v_hat[0, 1], v_hat[0, 0] - v_hat[1, 1])

                            #rotate_rect = ((est_cx, est_cy), (est_w*6, est_h*6), est_theta)
                            rotate_rect = ((est_cx, est_cy), (est_width*6, est_height*6), est_angle)
                            box = cv2.boxPoints(rotate_rect)

                            #second way of computing ellipse shape.
                            # _, est_size, phi = get_covariance_matrix_ellipse(v_hat)
                            # theta = np.real(phi)
                            # (lx, ly) = np.real(est_size)
                            #
                            # rotate_rect_fit = ((est_cx, est_cy), (lx * 12, ly * 12), theta * 180. / np.pi)
                            #box = cv2.boxPoints(rotate_rect)
                            canvas_frame = cv2.ellipse(canvas_frame, rotate_rect,(0,255,0),3)

                            obj_bbox = cv2.boundingRect(np.int0(box))
                        else:
                        #no blob selected, using prior_estimated information
                            #computing offset based on the last 4 frame's obj_bbox'center.
                            #using the average center shift as the (offset_x, offset_y)
                            if len(track_rect_list)<4:
                                prio_offset = (0,0)
                            else:
                                dif_rect = []
                                for iter in  [-1,-2,-3]:
                                    dif_rect.append(np.array(track_rect_list[iter]) - np.array(track_rect_list[iter-1]))
                                offset_rect = np.mean(dif_rect,0)
                                prio_offset = (offset_rect[0] + offset_rect[2]/2, offset_rect[1] + offset_rect[3]/2)
                                print( 'offset is (%d, %d)' %(prio_offset[0], prio_offset[1]))
                                prio_obj_bbox = (int(obj_bbox[0] + prio_offset[0]), int(obj_bbox[1] + prio_offset[1]), obj_bbox[2], obj_bbox[3])
                                obj_bbox = prio_obj_bbox
                        track_rect_list.append(obj_bbox)
                        uti.draw_rect(canvas_frame, obj_bbox, (0, 0, 255), 2)

                        total_time_consum = total_time_consum + time.clock() - tstart

                        bb_linedict = {}
                        tbb_key = 'frame ' + str(frame_id)
                        bb_linedict[tbb_key] = {'BoundingBox': obj_bbox}
                        tracker_tbbs_file.writelines(str(bb_linedict) + '\n')

                        # copy seg_roi to the canvas_frame
                        roih, roiw = seg_roi.shape[0:2]
                        canvas_frame[420:420 + roih, 770:770 + roiw, :] = seg_roi
                        uti.draw_rect(canvas_frame, (770, 420, roiw, roih), (255, 255, 255))

                    if frame_mode == 'polar':
                        cv2.imshow('polar', canvas_frame)
                        if frame_counter == 0:
                            cv2.moveWindow('polar', 0, 0)
                    else:
                        cv2.imshow('disp', canvas_frame)
                    cv2.waitKey(1)
                    frame_counter += 1.
                    img_res_name = '%s%s\\%04d.jpg' % (resPath,obj_name, frame_id)
                    #cv2.imwrite(img_res_name, canvas_frame)
            else:#readed file is empty
                break

            #total_time_consum = total_time_consum + time.clock() - tstart
        #closing dataset file
        fdata_obj.close()

    track_res_file.close()
    time_per_frame = total_time_consum * 1. / frame_counter
    print ('%s: average time consum per frame %fs \n' % (obj_name, time_per_frame))
    return time_per_frame


if __name__ == '__main__':

    #try to fix the debug mode confliction between matplotlib and cv2.waitKey()
    #from matplotlib import pyplot
    import matplotlib
    matplotlib.use('Agg')

    #pyplot.ioff()

    GtInfo = load_zhicheng_GT()

    roi_poly_name = '../ground_truth/roi_polygon/zhicheng_env.txt'
    roi_mask = uti.get_env_roi_mask(roi_poly_name, height=600, width=2048)
    # ParamOptionsDict = {'BlobSeg': {'MorphIterLst':[2], 'DilateIterLst':[3], 'DistThreashLst':[0.7, 0.6, 0.5, 0.4], 'SubWindow':[(128,64), (200,100)]},
    #                     'VotePsrLst':[5,8,10],
    #                     'VoteDistThreashLst':[90,100,120],
    #                     'PrioSigmaFactorLst':[1./4, 1./2, 3./4, 1., 1.1, 1.2],
    #                     'BlobScoreLst':[0.15, 0.2, 0.25, 0.3, 0.35]
    #                     }

    # ParamOptionsDefault= {'BlobSeg': {'MorphIter':2, 'DilateIter':3, 'DistThreash':0.6, 'SubWindow':(128,64)},
    #                     'VotePsr':5,
    #                     'VoteDistThreash':100,
    #                     'PrioSigmaFactor':3./4,
    #                     'BlobScore':0.2
    #                     }
    resPath = '../results/Res_EOTGGIW/'

    timeFile = open(resPath + 'cost_time.txt', 'w')
    ntarget = 0
    timeCounter = 0

    for key in GtInfo:
        # if key != 'Camen':
        #     continue
        # if key == 'Billy' or key =='Camen':
        #     continue
        #if  key =='Dolphin':
        #        continue
        #if key == 'Ellen': #and key != 'ALice':
        #     continue
        # if key != 'Camen':
        #       continue
        GtElem = GtInfo[key]
        paramFile = open(resPath+key+'_Params.txt', 'w')
        print ('Evaluate %s parameters: ' % key)

        # loading GtData and start_id, end_id
        GtDict = GtElem['GtData']
        start_id = GtElem['StartId']
        end_id = GtElem['EndId']


        ParamOptionsDefault= {'BlobSeg': {'MorphIter':2, 'DilateIter':3, 'DistThreash':0.6, 'SubWindow':(200,100)},
                            'VotePsr':10,
                            'VoteDistThreash':100,
                            'PrioSigmaFactor':3./4,
                            'BlobScore':0.15,
                            #'BlobScore': 0.3,
                            'NumberOfTrackers':10
                            }
        #BlobScore influence the tracker's shrinking speed.
        #for trial,number in enumerate([15, 10, 8, 5, 3]):
        for trial, number in enumerate([10]):
            #ParamOptionsDefault['PrioSigmaFactor'] = factor
            ParamOptionsDefault['NumberOfTrackers'] = number
            paramFile.writelines(str(ParamOptionsDefault)+'\n')
            paramFile.flush()
            time_per_frame = load_frame(key, trial, GtDict, roi_mask, start_id, end_id, ParamOptionsDefault)
            paramFile.writelines('Averaged_time_per_frame %f s\n' % time_per_frame)

            timeFile.writelines('%s averaged_time_per_frame %f s\n' % (key, time_per_frame))
            ntarget += 1.
            timeCounter += time_per_frame
    timeFile.writelines('Over_%d_targets average_time_per_frame %f s' % (ntarget, timeCounter / ntarget))
    timeFile.close()
