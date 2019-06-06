"""
The main entrance for the Multiple Kernelized Correlation Filters (MKCF),
The paper is published in IEEE TSP now (https://ieeexplore.ieee.org/document/8718392).
Source paper with supplementary materials (diagrams of the algorithm and curve results)
is also available on the following link:
(https://github.com/joeyee/MKCF/blob/master/MKCF_SourcePaper_SingleColumn.pdf)


The aim for MKCF is :
To enhance the long-term tracking, multiple trackers are fused to track a proper object.

load data from radar datasets file ->
     -> send one frame to the MKCF tracker
            -> initial frame with marked ROI initialize the first tracker
            -> later frame update the MKCF tracker, blob segmentation and multiple tracker rectangles involved voting
               give birth to new individual tracker if necessary.
            -> fusion multiple tracker via the maximum likelihood criterion.
            -> output the final tracker rectangle (position and width, height)

"""


import cv2
import pylab
pylab.ioff()   # try to avoid the conflict between pylab.imshow() and cv2.waitKey() on Mac
import time
import numpy               as np
import coordinates_convert as corcon
import KCFtracker_Status_MotionVector   as KCF_ST_MV
import utility as uti
from   PIL     import Image


def load_zhicheng_GT():
    '''
    Set the path and GT files name, according start_id, end_id.
    Convert the Gt file information into a dictionary data type [GtInfo]
    :return GtInfo:
    '''
    FilePath = '../ground_truth/'
    GtParam = {'ALice':{'FileName':'zhicheng20130808_22.xx_60-400_polarGT_Alice.txt',   'StartId':60, 'EndId':400},
              'Billy':{'FileName':'zhicheng20130808_22.xx_2-215_polarGT_Billy.txt',     'StartId':2,  'EndId':226},
              'Camen':{'FileName':'zhicheng20130808_22.xx_2-408_polarGT_Camen.txt',     'StartId':75,  'EndId':335},
              #'Camen': {'FileName': 'zhicheng20130808_22.xx_2-408_polarGT_Camen.txt', 'StartId': 2, 'EndId': 408},
              'Dolphin':{'FileName':'zhicheng20130808_22.xx_72-316_polarGT_Dolphin.txt','StartId':73, 'EndId':316},
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

def fuse_trackers(tracker_list):
    '''
    fuse all the trackers to get the estimated bounding box, bb.
    :param tracker_list:
    :return fused_bb:
    '''
    num_t = len(tracker_list)
    #1 compute the tlx, tly, brx, bry of the fused region
    tlx,tly = (10000000,1000000)
    brx,bry = (0,0)
    for tracker in tracker_list:
        if tlx > tracker.tlx : tlx = int(tracker.tlx)
        if tly > tracker.tly : tly = int(tracker.tly)
        if brx < tracker.tlx + tracker.window_sz[1]:  brx = int(tracker.tlx + tracker.window_sz[1])
        if bry < tracker.tly + tracker.window_sz[0]:  bry = int(tracker.tly + tracker.window_sz[0])

    fuse_layers = np.zeros(((bry-tly), (brx-tlx), num_t))
    for i in range(num_t):
        offset_x = int(tracker_list[i].tlx) - tlx
        offset_y = int(tracker_list[i].tly) - tly
        rw       = int(tracker_list[i].window_sz[1])
        rh       = int(tracker_list[i].window_sz[0])
        fuse_layers[offset_y:offset_y+rh, offset_x:offset_x+rw, i] = tracker_list[i].response
        '''
        draw multiple layers on the plotlib
        '''

    ml_matrix = np.zeros(((bry - tly), (brx - tlx)))
    for x in range(brx-tlx):
        for y in range(bry-tly):
            ml = 1.
            for i,tracker in enumerate(tracker_list):
                ofx = (tlx + x) - int(tracker.tlx)
                ofy = (tly + y) - int(tracker.tly)
                rw  = int(tracker.window_sz[1])
                rh  = int(tracker.window_sz[0])
                #using offset to locate ml value in the response matrix
                if (ofx>=0 and ofy>=0) and (ofx<rw and ofy<rh):
                    ml *= tracker.response[ofy, ofx]
                else: # out of tracker.response region, assign little value
                    ml *= np.spacing(1)
            ml_matrix[y,x] = ml

    # target location is at the maximum ml matrix
    est_y, est_x = pylab.unravel_index(ml_matrix.argmax(), ml_matrix.shape)

    # est_tlx = est_x - int(ml_matrix.shape[1]/2) + tlx
    # est_tly = est_y - int(ml_matrix.shape[0]/2) + tly

    # target scale is at the maximum response(est_y, est_x)
    ml_wh = np.spacing(1)
    est_w, est_h = 0,0
    for i,tracker in enumerate(tracker_list):
        ofx = (tlx + est_x) - int(tracker.tlx)
        ofy = (tly + est_y) - int(tracker.tly)
        rw = int(tracker.window_sz[1])
        rh = int(tracker.window_sz[0])
        # using offset to locate ml value in the response matrix
        if (ofx >= 0 and ofy >= 0) and (ofx < rw and ofy < rh):
            if(ml_wh < tracker.response[ofy, ofx]):
                ml_wh = tracker.response[ofy, ofx]
                est_h, est_w = int(tracker.target_sz[0]), int(tracker.target_sz[1])

    est_bb = [int((est_x - est_w/2) + tlx), int((est_y-est_h/2) + tly), est_w, est_h]
    return est_bb

def fuse_approx_trackers(trackerlist, vote_psr_threash=10):
    '''
    Assuming that the trackerlist contain's all the tracker which is greater or equal votePsrThreashold 10,
    then we can approximate the y'_i in Gaussian distribution, see the source paper [Section IV-C] for details.
    :param trackerlist:
    :return: fused bounding box of the obj_bbox.
    '''

    peak_list = []
    tbb_list  = []
    for i,tracker in enumerate(trackerlist):
        if tracker.psr >= vote_psr_threash:
            peak_list.append(tracker.responsePeak)
            tbb_list.append(tracker.rect)
    if len(tbb_list) == 0:
        print('Pay attention! No qualified tracker for fusing!!!')

    peaks = np.array(peak_list)
    tbbs  = np.array(tbb_list)
    weights = peaks**2
    weights = weights / np.sum(weights)
    weights = weights.reshape(len(weights), 1)
    weights = np.tile(weights, (1, 4))
    obj_bbox_array = np.int0(np.sum(tbbs * weights, axis=0))
    obj_bbox = (obj_bbox_array[0], obj_bbox_array[1], obj_bbox_array[2], obj_bbox_array[3])
    return obj_bbox


def weight_tbox(obj_rect, offset, tbbs, psrs, sigma_factor = 1./4):
    '''
    center means the gaussian prior's maximum position, offset should be add before this call
    sigma_w, sigma_h 2d gaussian has two sigmas for row and col
    tbbs, tracked_bounding_box'list to np.array.
    psrs, tbbs' psr score list to np.array.

    in the revised version, omit the prior
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

    # #position array
    # posxs = tbbs[:,0] + tbbs[:,2]/2
    # posys = tbbs[:,1] + tbbs[:,3]/2
    #
    # cx, cy = (obj_rect[0]+obj_rect[2]/2, obj_rect[1]+obj_rect[3]/2)
    # w      = obj_rect[2] * 2
    # h      = obj_rect[3] * 2
    #
    # sigma_w = sigma_factor*w
    # sigma_h = sigma_factor*h
    #
    # ofx, ofy = offset[0:2]
    # distxs = np.float32(posxs) -(cx + ofx)
    # distys = np.float32(posys) -(cy + ofy)
    #
    # #priority probability
    # pprob = pylab.exp(-0.5 * ((distxs / sigma_w) ** 2 + (distys / sigma_h) ** 2))
    #
    # psrs = psrs / (np.sum(psrs)+np.spacing(1))
    #
    # #NOTE, here the pprob are normalized before using!!! so it's not gaussian distribution anymore.
    # norm_pprob = pprob / (np.sum(pprob) + np.spacing(1))
    #
    # #weights = psrs * norm_pprob
    # #usnig psr as weights only, ignore the prior
    # weights = psrs
    #
    # weights = weights / np.sum(weights)
    # weights = norm_pprob.reshape(len(weights), 1)
    # weights = np.tile(weights, (1, 4))
    #
    # # obj_bbox is the average rect of all the tbbs with responding weights, given by tracker's psr and prio-probability
    # obj_bbox_array = np.int0(np.sum(tbbs * weights, axis=0))
    # obj_bbox = (obj_bbox_array[0], obj_bbox_array[1],obj_bbox_array[2], obj_bbox_array[3])


    #Revised version using psr^2/ sum(psr^2)
    weights = psrs**2
    weights = weights / np.sum(weights)
    weights = weights.reshape(len(weights), 1)
    weights = np.tile(weights, (1, 4))
    obj_bbox_array = np.int0(np.sum(tbbs * weights, axis=0))
    obj_bbox = (obj_bbox_array[0], obj_bbox_array[1],obj_bbox_array[2], obj_bbox_array[3])
    return obj_bbox

#For controlling the detailed display information.
DETAIL_MODE = False
def load_frame(obj_name, trial_times, GtDict, roi_mask, start_id, end_id, ParamOptions):
    """
    :param obj_name: the name of the interested object
    :param trial_times: repeating times
    :param GtDict: ground truth
    :param roi_mask:  map mask which omitting the echoes from the land and  bridge
    :param start_id: start frame id
    :param end_id:   end frame id
    :param ParamOptions: Key parameters for the MKCF
    :return:
    """
    """
    Get the formatted sub-image_list, append in polar or cart subimage list.
    data_name given the source binary data file name.
    frame_id, is indexed for several sequences in sequential time.
    GT_dict give all the ground truth information for a specific target
    roi_mask is the region of interesting for a tracking environment
    """
    print( 'Tracking %s:...' % obj_name)
    resPath = '../results/Res_MKCF/'
    #*_tbbs.txt for averaged boundingbox only
    #*_blob.txt for matched blob
    tracker_res_name  = resPath + obj_name + '_blobs_%02d_option.txt' % trial_times
    tracker_tbb_name  = resPath + obj_name + '_MKCF_Tbbs.txt'
    tracker_num_name  = resPath + obj_name + '_MKCF_Nums.txt'
    track_res_file    = open(tracker_res_name, 'w')
    tracker_tbbs_file = open(tracker_tbb_name, 'w')
    tracker_num_file  = open(tracker_num_name, 'w')

    # ParamOptions= {'BlobSeg': {'MorphIter':2, 'DilateIter':3, 'DistThreash':0.6, 'SubWindow':(128,64)},
    #                     'VotePsr':5,
    #                     'VoteDistThreash':100,
    #                     'PrioSigmaFactor':3./4,
    #                     'BlobScore':0.2
    #                     }
    segBlobParam     = ParamOptions['BlobSeg']
    VotePsrThreash   = ParamOptions['VotePsr']
    VoteDistThreash  = ParamOptions['VoteDistThreash']
    BlobScoreThreash = ParamOptions['BlobScore']
    PrioSigmaFactor  = ParamOptions['PrioSigmaFactor']
    NumberOfTrackers = ParamOptions['NumberOfTrackers']

    data_name = []
    frame     = np.array([])
    #polar frame in float32
    dframe    = np.array([])
    #polar frame in uint8
    uframe    = np.array([])
    #cartesian frame in float32
    dispmat   = np.array([])
    #udispmat frame in uint8
    udispmat  = np.array([])
    #pre and cur frame for flow computing
    preframe  = np.array([])
    curframe  = np.array([])

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
    total_avepsr_consum = 0
    total_avelife_consum= 0
    total_ttnums_consum = 0
    frame_mode = 'polar'

    # reading the sequential 4 files in a loop
    for i in np.arange(4):

        #please replace the path with yours.
        file_prefix = '/Users/yizhou/Radar_Datasets/Zhicheng/zhicheng_20130808_22.'
        file_suffix = '_5minutes_600x2048.double.data'
        data_name = '%s%d%s' % (file_prefix, 28 + i * 5, file_suffix)
        print(data_name)
        fdata_obj = open(data_name, 'rb')
        # loading file data and to convert from polar to cartesian
        while True:
            #tstart = time.clock()
            if frame_id > end_id:
                #release the frame and break the while loop
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
                    # cv2.putText(canvas_polar, obj_name+' frame ' + str(frame_id), org=(10, 50),
                    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                    #             thickness=2, lineType=cv2.LINE_AA)
                    #cv2.imshow('polar', canvas_polar)

                    # for cartesian-coordinates
                    dispmat = corcon.polar2disp_njit(dframe, np.array([]))
                    dispmat = uti.frame_normalize(dispmat)
                    udispmat = (dispmat * 255).astype(np.uint8)
                    canvas_disp = cv2.cvtColor(udispmat, cv2.COLOR_GRAY2BGR)
                    cv2.putText(canvas_disp, 'frame ' + str(frame_id), org=(10, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                                thickness=2, lineType=cv2.LINE_AA)
                    #cv2.imshow('disp', canvas_disp)
                    # cv2.moveWindow('polar',0,0)
                    cv2.waitKey(1)

                    if frame_mode == 'polar':
                        #omitting the land and bridge echoes by '&' roi_mask.
                        tframe  = dframe * roi_mask
                        canvas_frame = canvas_polar
                    else:
                        tframe  = dispmat
                        tuframe = udispmat
                        canvas_frame = canvas_disp

                    if frame_id == start_id :

                        obj_bbox = objGtElem['BoundingBox']
                        # Initialize tracker with first frame and bounding box
                        tracker = KCF_ST_MV.KCFTracker_status()
                        ok = tracker.init(tframe, obj_bbox)
                        if ok:
                            tracker_list.append(tracker)


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
                        #timer = uti.msTimer()
                        #ms_start = timer.start()
                        # if(frame_id == 106):
                        #     print(frame_id)

                        print('::frame id %d\t, tracker nums %d -of- %d maxnums' % (frame_id, len(tracker_list), NumberOfTrackers))

                        tstart = time.clock()
                        #seg window is initialized by last frame's boundingbox
                        blob_list, seg_roi, sub_blob_list = uti.blob_seg(tframe, obj_bbox,
                                                                         segBlobParam['MorphIter'], segBlobParam['DilateIter'], segBlobParam['DistThreash'],
                                                                         segBlobParam['SubWindow'])

                        ##
                        #tracking results for high psr output trackers
                        #monitoring the tracker's status in PSR and distance to the fused obj_bbox.
                        ##
                        high_psr_tbox_list = []
                        votable_tbox_list = []
                        psr_list  = []
                        votable_psr_list = []
                        votable_tracker_id = []
                        for i,tracker in enumerate(tracker_list):
                            ok, bbox, psr, response = tracker.update(tframe)
                            dist2obb = np.sqrt(( bbox[0]+bbox[2]/2 - obj_bbox[0] - obj_bbox[2]/2)**2
                                              +( bbox[1]+bbox[3]/2 - obj_bbox[1] - obj_bbox[3]/2)**2)
                            #monitoring the status of the trackers
                            tracker.status_monitor(psr, dist2obb, psr_threash=10, dist_theash=100)

                            if (DETAIL_MODE == True):
                                print('tracker...psr %2.1f, life %d' % (psr, tracker.trackNo))
                            #only psr is higher enough, tbbs add to the votable_tbb list to vote for the new blob
                            if(psr > VotePsrThreash):
                                high_psr_tbox_list.append(bbox)
                                psr_list.append(psr)
                                p1 = (int(bbox[0]), int(bbox[1]))
                                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                                #draw less light yellow for high_psr_tracker's bb
                                cv2.rectangle(canvas_frame, p1, p2, (0, 100, 100),2)
                                cv2.putText(canvas_frame, str(i), org=p1,
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 125, 0),
                                            thickness=1, lineType=cv2.LINE_AA)
                                #add votalbe bbox based on the dist to the obj_box center less than the distThresh
                                if dist2obb <= VoteDistThreash:
                                    votable_tbox_list.append(bbox)
                                    votable_psr_list.append(psr)
                                    votable_tracker_id.append(i)
                                    # draw highlight yellow for votable boundingbox
                                    uti.draw_rect(canvas_frame, bbox, (0, 255, 255), 1)
                                    #if (DETAIL_MODE == True):
                                    #    print('....voting tracker...psr%d, life %d' % (psr, tracker.trackNo))
                            #drawing desgrade trackers' bb
                            else:
                                #tracker' tbb is drawn in GRAY, if it has no right to vote!
                                uti.draw_rect(canvas_frame, bbox, (125,125,125),1)
                                ptL = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
                                #ptb = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]))
                                cv2.putText(canvas_frame, '%d-%d-%d'%(int(psr),np.sum(tracker.highpsr_container), np.sum(tracker.collabor_container)), org=ptL,
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255),
                                            thickness=1, lineType=cv2.LINE_AA)

                        ##
                        #When the trackerlist is full, len(trackerlist) >= NumberOfTrackers,
                        #Delete the tracker if it is not votable (psr < votepsrthreash).
                        #Dynamically changing (+1 or -1) the NumberOfTrackers, according to the ave_psr in the trackerlist.
                        #
                        ##
                        minpsr = 10000
                        maxlife = 0
                        minlife = 10000
                        ave_psr = 0
                        ave_life = 0

                        # find the min_psr and its tracker object
                        # compute the ave_psr and ave_life
                        for tracker in tracker_list:
                            if (DETAIL_MODE == True):
                                print('life-%d, psr-%d, continue_psr-%d, size:%s' \
                                      % (tracker.trackNo, tracker.psr, np.sum(tracker.highpsr_container),
                                         str(tracker.target_sz)))
                            ave_psr += tracker.psr
                            ave_life += tracker.trackNo
                            if tracker.psr < minpsr:
                                minpsr = tracker.psr
                                minpsr_tracker = tracker
                            if tracker.trackNo > maxlife:
                                maxlife = tracker.trackNo
                            if tracker.trackNo < minlife:
                                minlife = tracker.trackNo
                        # increase the numberofTrackers only when the ave_psr is below 15 (vote_psr is 10)
                        ave_psr = ave_psr * 1. / len(tracker_list)
                        ave_life = ave_life * 1. / len(tracker_list)
                        if DETAIL_MODE == True: print('Average PSR %2.2f, Average LIFE %3.1f' % (ave_psr, ave_life))

                        if len(tracker_list) >= NumberOfTrackers:
                            # if minpsr >= VotePsrThreash+10 and minlife >= 20: #25 for dolphin
                            #     tracker_list.remove(minpsr_tracker)
                            if minpsr < VotePsrThreash:
                                tracker_list.remove(minpsr_tracker)
                                if DETAIL_MODE == True:
                                    print('--Del--minpsr life-%d, psr-%d, continue_psr-%d, size:%s' \
                                          % (minpsr_tracker.trackNo, minpsr_tracker.psr,
                                             np.sum(minpsr_tracker.highpsr_container),
                                             str(minpsr_tracker.target_sz)))

                        if ave_psr <= VotePsrThreash:  # increase trackernumber.
                            NumberOfTrackers = min(NumberOfTrackers + 1, 5)
                            if DETAIL_MODE == True: print(
                                'Average PSR is below %d, increasing one for maximum' % VotePsrThreash)

                        # decrease tracker number, minimum is equal to the initial parameter settings.
                        if ave_psr > VotePsrThreash:
                            NumberOfTrackers = max(NumberOfTrackers - 1, ParamOptions['NumberOfTrackers'])
                            if DETAIL_MODE == True:
                                if NumberOfTrackers != ParamOptions['NumberOfTrackers']: print('Decrease one tracker')


                        if len(high_psr_tbox_list)!=0:
                            fuse_approx_time = time.clock()
                            obj_bbox = fuse_approx_trackers(tracker_list)
                            fuse_approx_time = time.clock() - fuse_approx_time
                            #print('fuse_approx_time %f', fuse_approx_time)
                            # computing offset based on the last 4 frame's obj_bbox'center.
                            # using the average center shift as the (offset_x, offset_y)
                            if len(track_rect_list)>= 4:
                                dif_rect = []
                                for iter in [-1, -2, -3]:
                                    dif_rect.append(np.array(track_rect_list[iter]) - np.array(track_rect_list[iter - 1]))
                                offset_rect = np.mean(dif_rect, 0)
                                #prio_offset = (offset_rect[0] + offset_rect[2] / 2, offset_rect[1] + offset_rect[3] / 2)
                                prio_offset = (2*offset_rect[0] + offset_rect[2], 2*offset_rect[1] + offset_rect[3])
                        else:#this is only happens, when across the bridge.
                            print('All the trackers are degrading!!! (1)Using nearest blob bounding box or (2)Inertial navigation begins!')

                            if len(tracker_list)<=4:
                                prio_offset = (0,0)
                            print('offset value ', prio_offset)

                            '''
                            (1)Take the nearest blob's bounding box for all the tracker's new position, which is similiar to Kalmanfilter or EOT.
                            (2)If there is no blob exists in near by. Using inertial navigation.
                            '''
                            dist_min = 200
                            dist = 0
                            cx = obj_bbox[0] + obj_bbox[2] / 2
                            cy = obj_bbox[1] + obj_bbox[3] / 2
                            near_blob_rect = []
                            for blob in blob_list:
                                (bcx, bcy) = blob['Centroid']
                                dist = np.sqrt((cx-bcx)**2+(cy-bcy)**2)
                                if dist < dist_min:
                                    dist_min = dist
                                    near_blob_rect = blob['BoundingBox']

                            if dist_min < 200 :
                                #using the nearest blob's bounding box to guide the tracker's location, similiar to the KF and EOT trackers.
                                #This conditions is desinged only for the bridge crossing, when the target is all occluded by the bridge.
                                obj_bbox = near_blob_rect
                                print('...Please note, using nearest blob!!!...')
                            else:
                                #if no blob exists, using intertial navigation
                                pre_bbox = track_rect_list[-1]
                                #maintain last position + estimated Prio_offset
                                obj_bbox = (pre_bbox[0] + int(prio_offset[0]), pre_bbox[1] + int(prio_offset[1]), pre_bbox[2], pre_bbox[3])
                            for tracker in tracker_list:
                                    tracker.refresh_position(obj_bbox)


                        if(DETAIL_MODE==True):
                            print ('voting trackers - all trackers %d-%d' % (len(votable_tbox_list), len(tracker_list)))


                        #voted object blob
                        obj_blob = {}
                        #if len(tracker_list) < NumberOfTrackers:
                            #getting the related blob and using the current blob's bounding box to initial the newtracker
                        obj_blob, blob_score,blob_id  = uti.vote_blob(blob_list, votable_tbox_list)
                            #checking's the selected blob's score, to dicide whethear init new tracker or not
                            #if blob's score is average lower in all trackers, no init.
                            # obj_bbox should based on the average of all voted Trackers

                        using_blob_for_tracker = False
                        if obj_blob!={}:
                            blob_bb = obj_blob['BoundingBox']
                            if len(tracker_list) < NumberOfTrackers:
                            #only the blob_score enough high, new tracker is initialized
                                if (blob_score > BlobScoreThreash):
                                    #newtracker = KCF_ST.KCFTracker_status()
                                    newtracker = KCF_ST_MV.KCFTracker_status()
                                    #newtracker = KCF_ST_MV_LK.KCFTracker_status()
                                    ok = newtracker.init(tframe, blob_bb)
                                    if ok:
                                        tracker_list.append(newtracker)
                                        if (DETAIL_MODE == True): print('Adding a new tracker')
                                else:
                                    print('Voted blob is not qualified!')

                            blob_tt_iou = uti.intersection_rect(blob_bb, obj_bbox)
                            print('blob bb intersected with obj_bbox %.2f' % blob_tt_iou)
                            if (DETAIL_MODE == True): print('blob bb intersected with obj_bbox %1.2f' % blob_tt_iou)
                            ##using measured blob's bounding box to improve the tracking results.
                            # if obj_box contains the voted blob, we use the blob to improve the results
                            #if uti.is_rect_contained(blob_bb, obj_bbox):
                            if uti.is_rect_contained(blob_bb, obj_bbox):
                                using_blob_for_tracker = True
                        else:#voted blob is null
                            print('No blob is voted!')

                        if using_blob_for_tracker:
                            rotate_rect =  obj_blob['RotatedBox']
                            track_rect_list.append(blob_bb)
                        else:
                            cx = int((obj_bbox[0] + obj_bbox[2]/2))
                            cy = int((obj_bbox[1] + obj_bbox[3]/2))
                            w  = obj_bbox[2]
                            h = obj_bbox[3]
                            if h>=w:
                                theta = 180
                            else:
                                theta = 0
                            rotate_rect    = ((cx, cy), (w, h), theta)
                            #rotate_rect = obj_blob['RotatedBox']
                            track_rect_list.append(obj_bbox)

                        #draw tracked boundingbox in rotated_rect form
                        canvas_frame = cv2.ellipse(canvas_frame, rotate_rect, (0, 255, 0), 1)
                        #uti.draw_rect(canvas_frame, track_rect_list[-1], (0, 255, 0),3)  # blodest green for confirmed measurement
                        uti.draw_rect(canvas_frame, obj_bbox, (0, 0, 255), 2) # bold red for tracker's fusion

                        #uti.draw_rect(canvas_frame, fuse_obj_bbox, (255,255,255), 2) # draw ml fused rectangle

                        frame_counter += 1.
                        total_time_consum = total_time_consum + time.clock() - tstart

                        total_avepsr_consum += ave_psr
                        total_avelife_consum+= ave_life
                        total_ttnums_consum += len(tracker_list)
                        if (DETAIL_MODE == True):
                            print ('==frame id %d\t, cost %.4f seconds\n' % (frame_id, (time.clock() - tstart)))
                            #print ('frame %d cost %.4f seconds' % (frame_id, mscost))

                        #drawing the subblob
                        if obj_blob!={}:
                            blob_bb  = obj_blob['BoundingBox']
                            #draw voted blob in blue color.
                            uti.draw_rect(canvas_frame, blob_bb, (255, 0, 0),2)

                            # sub_blob in seg_roi coordinates sharing the same blob_id in blob_list.
                            sub_blob = sub_blob_list[blob_id]
                            cnt = np.array([sub_blob['Polygon']])
                            #Draw the selected blob in yellow color.
                            cv2.polylines(seg_roi, cnt, True, (0, 255, 255), 3)


                            cv2.putText(canvas_frame, str('%0.3f'% blob_score), org=(740,420),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0, 255, 0),
                                        thickness=1, lineType=cv2.LINE_AA)
                            #voted average overlapped score, for checking the blob_seg function's effect.
                            obj_blob['BlobScore'] = blob_score
                            blob_key = 'frame ' + str(frame_id)
                            linedict = {}
                            linedict[blob_key] = obj_blob
                            track_res_file.writelines(str(linedict) + '\n')
                            track_res_file.flush()


                        bb_linedict = {}
                        tbb_key = 'frame ' + str(frame_id)
                        #bb_linedict[tbb_key] = {'BoundingBox': obj_bbox}
                        bb_linedict[tbb_key] = {'BoundingBox': track_rect_list[-1]}
                        tracker_tbbs_file.writelines(str(bb_linedict) + '\n')

                        mtrackers_status = {}
                        mtrackers_status[tbb_key] = {'ave_psr': ave_psr.__format__('.4'), 'min_psr': minpsr.__format__('.4'), 'ave_life': ave_life.__format__('.4'), 'tnums': len(tracker_list)}
                        tracker_num_file.writelines(str(mtrackers_status)+'\n')

                        str_msts = ' ave_psr %2.2f, min_psr %2.2f, ave_life %2.2f, tracker_nums %d' % (ave_psr, minpsr, ave_life, len(tracker_list))

                        cv2.putText(canvas_frame, obj_name + ' frame ' + str(frame_id) + str_msts, org=(10, 50),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                                    thickness=2, lineType=cv2.LINE_AA)

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
                    #print('tracker nums %d' % len(tracker_list))
                    img_res_name = '%s%s/%04d.png' % (resPath, obj_name, frame_id)
                    #saving image with high dpi
                    cvt_frame = cv2.cvtColor(canvas_frame, cv2.COLOR_BGR2RGB)
                    save_frame = Image.fromarray(cvt_frame)
                    #save_frame.save(img_res_name, dpi=(300, 300), compress_level=0)
                    #cv2.imwrite(img_res_name, canvas_frame)
            else:#readed file is empty
                break
            #frame_counter += 1.
            #total_time_consum = total_time_consum + time.clock() - tstart
        #closing dataset file
        fdata_obj.close()

    track_res_file.close()
    time_per_frame = total_time_consum / frame_counter
    psr_per_frame  = total_avepsr_consum / frame_counter
    life_per_frame = total_avelife_consum / frame_counter
    ttnums_per_frame = total_ttnums_consum / frame_counter

    str_stats = ''
    str_stats.join('%s: average time consum per frame \t  %.5fs \n'  % (obj_name, time_per_frame))
    str_stats.join('%s: average psr score per frame \t    %2.2f \n' % (obj_name, psr_per_frame))
    str_stats.join('%s: average life  per frame \t        %2.2f \n' % (obj_name, life_per_frame))
    str_stats.join('%s: average ttnums  per frame \t      %2.2f \n' % (obj_name, ttnums_per_frame))
    print(str_stats)
    return time_per_frame, str_stats


if __name__ == '__main__':
    import matplotlib
    #matplotlib.use('TkAgg')
    matplotlib.use('Agg')
    pylab.ioff()

    # cv2.namedWindow('polar', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('polar', 1000, 800)
    cv2.namedWindow('polar')
    cv2.moveWindow('polar', 0, 0)

    #load ground truth file in GtInfo dictionary.
    GtInfo = load_zhicheng_GT()

    #load map mask (polygon vertex) into the roi_mask
    roi_poly_name = '../ground_truth/roi_polygon/zhicheng_env.txt'
    roi_mask = uti.get_env_roi_mask(roi_poly_name, height=600, width=2048)
    resPath = '../results/Res_MKCF/'

    timeFile = open(resPath + 'cost_time.txt', 'w')
    ntarget = 0
    timeCounter = 0

    #first viersion using distTreash 0.6, currently using 0.3
    #'BlobSeg' is the parameter for the segmenation in OpenCV
    ParamOptionsDefault = {'BlobSeg': {'MorphIter': 2, 'DilateIter': 3, 'DistThreash': 0.3, 'SubWindow': (200, 100)},
                           #VotePsr named Sth in Table IV of the source paper
                           'VotePsr': 10,  #The minimum PSR scores for the qualified tracker.
                           'VoteDistThreash': 100,  # Votable distance limitation, ensuring the votalbe trackers are from the neighbourhood.
                           'PrioSigmaFactor': 3. / 4, #out of use
                           'BlobScore': 0.15,      #Oth in the paper
                           'NumberOfTrackers': 3   #number of the fused trackers
                           }
    for key in GtInfo:
        GtElem = GtInfo[key]
        paramFile = open(resPath+key+'_Params.txt', 'w')
        print ('Evaluate %s parameters: ' % key)

        # loading GtData and start_id, end_id
        GtDict = GtElem['GtData']
        start_id = GtElem['StartId']
        end_id = GtElem['EndId']

        #BlobScore influence the tracker's shrinking speed.
        #for trial,number in enumerate([15, 10, 8, 5, 3]):
        #Here can try different  NumberOfTrackers parameter.
        for trial, number in enumerate([3]):
            #ParamOptionsDefault['PrioSigmaFactor'] = factor
            ParamOptionsDefault['NumberOfTrackers'] = number
            paramFile.writelines(str(ParamOptionsDefault)+'\n')
            paramFile.flush()
            time_per_frame, str_statistics = \
                load_frame(key, trial, GtDict, roi_mask, start_id, end_id, ParamOptionsDefault)

            paramFile.writelines(str_statistics)

            timeFile.writelines('%s averaged_time_per_frame %f s\n' % (key, time_per_frame))
            ntarget += 1.
            timeCounter += time_per_frame
    timeFile.writelines('Over_%d_targets average_time_per_frame %f s' % (ntarget, timeCounter / ntarget))
    timeFile.close()
    paramFile.close()
