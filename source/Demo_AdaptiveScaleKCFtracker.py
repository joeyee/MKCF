#!/usr/bin/python
# -*- coding: utf-8 -*-

# Test the adaptive scale performance

import cv2

import time

import numpy               as np
import coordinates_convert as corcon

import KCFtracker_AdaptiveScales        as KCF_ADAP_SCALE
import utility             as uti



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

#For controlling the detailed display information.
DETAIL_MODE = False
def load_frame(obj_name, trial_times, GtDict, roi_mask, start_id, end_id, ParamOptions):
    """
    Get the formatted sub-image_list, append in polar or cart subimage list.
    data_name given the source binary data file name.
    frame_id, is indexed for several sequences in sequential time.
    GT_dict give all the ground truth information for a specific target
    roi_mask is the region of interesting for a tracking environment
    """
    print ('Tracking %s:...' % obj_name)
    resPath = '../results/Res_ASKCF/'
    #*_tbbs.txt for averaged boundingbox only
    #*_blob.txt for matched blob
    tracker_res_name  = resPath + obj_name + '_blobs_%02d_option.txt' % trial_times
    tracker_tbb_name  = resPath + obj_name + '_ASKCF_Tbbs.txt'
    track_res_file    = open(tracker_res_name, 'w')
    tracker_tbbs_file = open(tracker_tbb_name, 'w')

    # ParamOptions= {'BlobSeg': {'MorphIter':2, 'DilateIter':3, 'DistThreash':0.6, 'SubWindow':(128,64)},
    #                     'VotePsr':5,
    #                     'VoteDistThreash':100,
    #                     'PrioSigmaFactor':3./4,
    #                     'BlobScore':0.2
    #                     }
    segBlobParam  = ParamOptions['BlobSeg']
    VotePsrThreash= ParamOptions['VotePsr']
    VoteDistThreash = ParamOptions['VoteDistThreash']
    BlobScoreThreash= ParamOptions['BlobScore']
    PrioSigmaFactor = ParamOptions['PrioSigmaFactor']
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

    frame_mode = 'polar'

    # reading the sequential 4 files in a loop
    for i in np.arange(4):
        file_prefix = '/Users/yizhou/Radar_Datasets/Zhicheng/zhicheng_20130808_22.'
        file_suffix = '_5minutes_600x2048.double.data'
        data_name = '%s%d%s' % (file_prefix, 28 + i * 5, file_suffix)
        print (data_name)
        fdata_obj = open(data_name, 'rb')
        # loading file data and to convert from polar to cartesian
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
                        tframe  = dframe * roi_mask
                        #echoThreash plays a role of CFAR.
                        echoThreash = np.mean(tframe)
                        tuframe = uframe
                        canvas_frame = canvas_polar
                    else:
                        tframe  = dispmat
                        tuframe = udispmat
                        canvas_frame = canvas_disp


                    if frame_id == start_id :
                        # init previous frame
                        preframe = tuframe
                        obj_bbox = objGtElem['BoundingBox']  # frame60 Alice
                        # Uncomment the line below to select a different bounding box
                        # bbox = cv2.selectROI(frame, False)
                        # Initialize tracker with first frame and bounding box
                        #tracker = KCF_ST.KCFTracker_status()

                        tracker = KCF_ADAP_SCALE.KCFTracker_AdaptiveScales()
                        ok = tracker.init(tframe, obj_bbox)

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
                        tstart = time.clock()

                        #seg window is initialized by last frame's boundingbox
                        blob_list, seg_roi, sub_blob_list = uti.blob_seg(tframe, obj_bbox,
                                                                         segBlobParam['MorphIter'], segBlobParam['DilateIter'], segBlobParam['DistThreash'],
                                                                         segBlobParam['SubWindow'])

                        cx = obj_bbox[0] + obj_bbox[2]/2
                        cy = obj_bbox[1] + obj_bbox[3]/2
                        dist_min = 200
                        dist = 0
                        blob_id = 0
                        for bid, blob in enumerate(blob_list):
                            (bcx,bcy) = blob['Centroid']
                            dist = np.sqrt((cx-bcx)**2+(cy-bcy)**2)
                            if dist < dist_min:
                                dist_min = dist
                                blob_id = bid
                        if dist_min < 200 and len(blob_list)>0:

                            ok, bbox, psr = tracker.update(tframe)
                            obj_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])

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

                        #mscost = timer.count()
                        frame_counter += 1.
                        total_time_consum = total_time_consum + time.clock() - tstart
                        if (DETAIL_MODE == True):
                            print ('frame %d cost %.4f seconds' % (frame_id, (time.clock() - tstart)))
                            print ('frame %d cost %.4f seconds' % (frame_id, mscost))

                        uti.draw_rect(canvas_frame, obj_bbox, (0, 0, 255), 2)


                        bb_linedict = {}
                        tbb_key = 'frame ' + str(frame_id)
                        bb_linedict[tbb_key] = {'BoundingBox':obj_bbox}
                        tracker_tbbs_file.writelines(str(bb_linedict) + '\n')


                        # copy seg_roi to the canvas_frame
                        roih, roiw = seg_roi.shape[0:2]
                        canvas_frame[420:420 + roih, 770:770 + roiw, :] = seg_roi
                        uti.draw_rect(canvas_frame, (770, 420, roiw, roih), (255, 255, 255))

                    if frame_mode == 'polar':
                        cv2.imshow('polar', canvas_frame)
                        if frame_counter == 1:
                            cv2.moveWindow('polar', 0, 0)
                    else:
                        cv2.imshow('disp', canvas_frame)

                    cv2.waitKey(1)
                    img_res_name = '%s%s\\%04d.jpg' % (resPath,obj_name, frame_id)
                    #cv2.imwrite(img_res_name, canvas_frame)
            else:#readed file is empty
                break
            #frame_counter += 1.
            #total_time_consum = total_time_consum + time.clock() - tstart
            #print '%s: average time consum per frame %fs \n' % (obj_name, time_per_frame)
            # closing dataset file
        fdata_obj.close()

    track_res_file.close()
    time_per_frame = total_time_consum / frame_counter
    print ('%s: average time consum per frame %fs \n' % (obj_name, time_per_frame))
    return time_per_frame


if __name__ == '__main__':
    #try to fix the debug mode confliction between matplotlib and cv2.waitKey()
    #from matplotlib import pyplot
    import matplotlib
    matplotlib.use('Agg')

    GtInfo = load_zhicheng_GT()

    #Mask the bridge echo.
    roi_poly_name = '../ground_truth/roi_polygon/zhicheng_env.txt'
    roi_mask = uti.get_env_roi_mask(roi_poly_name, height=600, width=2048)
    #resPath = 'F:\\code\\Python\\visual_radar_tracker\\images\\zhicheng20130808_22.28\\Res_KF\\'
    resPath = '../results/Res_ASKCF/'

    timeFile = open(resPath + 'cost_time.txt', 'w')
    ntarget = 0
    timeCounter = 0

    for key in GtInfo:
        # if key != 'Billy':# or key =='Camen':
        #      continue
        # if  key !='Dolphin':
        #       continue
        # if key != 'Ellen' and key != 'ALice':
        #     continue
        # if key != 'Camen':
        #      continue
        GtElem = GtInfo[key]
        paramFile = open(resPath+key+'_Params.txt', 'w')
        print ('Evaluate %s parameters: ' % key)

        # loading GtData and start_id, end_id
        GtDict = GtElem['GtData']
        start_id = GtElem['StartId']
        end_id = GtElem['EndId']


        ParamOptionsDefault= {'BlobSeg': {'MorphIter':2, 'DilateIter':3, 'DistThreash':0.6, 'SubWindow':(200,100)},
                            'VotePsr':5,
                            'VoteDistThreash':100,
                            'PrioSigmaFactor':3./4,
                            'BlobScore':0.15,
                            'NumberOfTrackers':10
                            }
        #BlobScore influence the tracker's shrinking speed.
        #for trial,number in enumerate([15, 10, 8, 5, 3]):
        for trial, number in enumerate([15]):
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