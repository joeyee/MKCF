#!/usr/bin/python
# -*- coding: utf-8 -*-

# Evaluate Five trackers: Kalman Filter, Extended Object Tracking based on Random Matrix(EOTRM),
# Kernelized Correlated Filter(KCF), Adaptive Scales KCF(ASKCF), Multiple KCFs(MKCF).
# Add EOT with GGIW prior (EOTGGIW) and Multi-kernel learning based KCF (MKL) for comparison
# Four Evaluation criteria are used to testify the effectiveness of the proposed Trackers.
# 1. Center location Error.  CLE is the difference between the center of tracked object and the ground truth,
#    where the smaller value means the more accurate result.
# 2. Intersection over Union (IoU). Given two shapes, IoU is the intersected area divided by their union area.
#    The larger value of IoU represents the more accurate result.
# 3. Precision plot based on CLE, The plots show, for a range of distance thresholds,
#    the percentage of frames that the tracker is within that distance of the ground truth
# 4. Precision plot based on IoU, The plots show, for a range of overlapped ratio thresholds,
#    the percentage of frames that the tracking IoU is larger than the defined ratio threshold.
import cv2
import numpy as np
import pylab
pylab.ioff()
import utility as uti


#TODO 跟踪位置与gt位置的差距，及其不同门限下的正确跟踪率统计（均值，方差，ROC面积） done
#TODO 跟踪矩形与GTblob的相交得分，将其与gtbb与gtblob的相交得分做比较。直接代入Gtdata到show_tbb_precesion即可。
#TODO 考虑通过单KCF中运动方向在psr低值下的稳定设置，来避免遮挡跑偏的问题。减少跟踪器数量。
#TODO 比较方法：MOSSE, KCF, EKF, PF, PHP[最后选择的是KF，EOT_RM, KCF, SAKCF, MKCF]
#TODO 171222开始写自己的work_flow和一些图像的素材准备。
def show_blob_precision(tt_dict, gt_dict):
    '''
    :param tt_dict: tracked object information in dict format
    :param gt_dict: ground truth information in dict format
    :return 4 evaluated results, and frame ids. fids.
    center location error (position dist),
    IoU(rect_intersection),
    precision plots on position distance, (posistion_precision)
    precision plot  on IoU,(rect_intersection_precision)
    '''
    gt_pos_list = []
    tt_pos_list = []
    gt_bbox_list = []
    tt_bbox_list = []
    gt_poly_list = []
    tt_poly_list = []
    # change the dictionary order to sequential(1,2,3,....)
    tfids = []
    for ttkey in tt_dict:
        tfid = int(ttkey.split(' ')[1])
        tfids.append(tfid)
    tfids.sort()

    for tfid in tfids:
        ttkey = 'frame %d' % tfid
        if ttkey not in gt_dict:
            continue
        tt_info_dict = tt_dict[ttkey]
        gt_info_dict = gt_dict[ttkey]
        if tt_info_dict != {} and gt_info_dict != {}:
            gt_pos_list.append(gt_info_dict['Center'])
            tt_pos_list.append(tt_info_dict['Center'])
            gt_bbox_list.append(gt_info_dict['BoundingBox'])
            tt_bbox_list.append(tt_info_dict['BoundingBox'])
            gt_poly_list.append(gt_info_dict['Polygon'])
            tt_poly_list.append(tt_info_dict['Polygon'])
    #print 'compared tracking numbers %d \n' % len(tt_pos_list)

    gt_poses = np.array(gt_pos_list, dtype=float)
    tt_poses = np.array(tt_pos_list, dtype=float)

    pos_dist = np.sqrt((gt_poses[:, 0] - tt_poses[:, 0]) ** 2 + (gt_poses[:, 1] - tt_poses[:, 1]) ** 2)
    pos_precision = np.zeros((100, 1), np.float)
    for thresh in range(100):
        tracked_nums = np.sum(pos_dist <= thresh, dtype=float)
        pos_precision[thresh] = tracked_nums / len(pos_dist)

    # the X_axis is the threashold and the y_axis is the precision.
    pylab.figure()
    pylab.plot(pos_precision)
    pylab.figure()
    pylab.plot(tfids, pos_dist)

    #computing the rectangle overlapping
    rect_inter_ratio = np.zeros((len(tt_bbox_list),1), np.float)
    for i,(tbb,gbb) in enumerate(zip(tt_bbox_list,gt_bbox_list)):
        rect_inter_ratio[i] = uti.intersection_rect(tbb,gbb)

    rint_threash_range = np.arange(1,0,-0.01).tolist()
    rect_precision = np.zeros((len(rint_threash_range), 1), np.float)
    for i,rint_threash in enumerate(rint_threash_range):
        tracked_nums = np.sum(rect_inter_ratio >= rint_threash, dtype=float)
        rect_precision[i] = tracked_nums / len(tt_bbox_list)

    pylab.figure()
    pylab.plot(rint_threash_range,rect_precision)
    pylab.figure()
    pylab.plot(tfids,rect_inter_ratio)

    #computing the polygon region overlapping
    # poly_inter_ratio = np.zeros((len(tt_poly_list), 1), np.float)
    # for i, (tpoly, gpoly) in enumerate(zip(tt_poly_list, gt_poly_list)):
    #     poly_inter_ratio[i] = uti.intersection_poly(np.array([tpoly]), np.array([gpoly]))
    #
    # pint_threash_range = np.arange(1, 0, -0.01).tolist()
    # poly_precision = np.zeros((len(pint_threash_range), 1), np.float)
    # for i, pint_threash in enumerate(pint_threash_range):
    #     tracked_nums = np.sum(poly_inter_ratio >= pint_threash, dtype=float)
    #     poly_precision[i] = tracked_nums / len(tt_bbox_list)
    #
    # pylab.figure()
    # pylab.plot(pint_threash_range,poly_precision)
    # pylab.figure()
    # pylab.plot(tfids,poly_inter_ratio)
    # pylab.show()
    return pos_dist, rect_inter_ratio, pos_precision, rect_precision, tfids

def show_tbb_precision(tt_dict, gt_dict, bplot = False):
    '''
    center point's distance and tracked bounding box(tbb) overlapped with ground truth polygon
    tbb overlapped gt_polygon, is similar to the blob_score, which is tbb overlapped with seg_blob' polygon
    :param tt_dict:
    :param gt_dict:
    :return 4 evaluated results, and frame ids. fids.
    center location error (position dist),
    IoU(rect_intersection),
    precision plots on position distance, (Position_precision)
    precision plot  on IoU,(rect_intersection_precision)
    '''
    gt_pos_list = []
    tt_pos_list = []
    gt_bbox_list = []
    tt_bbox_list = []
    gt_poly_list = []
    tt_poly_list = []
    # change the dictionary order to sequential(1,2,3,....)
    tfids = []
    for ttkey in tt_dict:
        tfid = int(ttkey.split(' ')[1])
        tfids.append(tfid)
    tfids.sort()

    #compared frame_id with both gt and tt.
    cpFids = []
    for tfid in tfids:
        ttkey = 'frame %d' % tfid
        if ttkey not in gt_dict:
            continue
        tt_info_dict = tt_dict[ttkey]
        gt_info_dict = gt_dict[ttkey]
        if tt_info_dict != {} and gt_info_dict != {}:
            gt_pos_list.append(gt_info_dict['Center'])
            gt_bbox_list.append(gt_info_dict['BoundingBox'])
            gt_poly_list.append(gt_info_dict['Polygon'])

            tbb = tt_info_dict['BoundingBox']
            center = (tbb[0] + tbb[2]/2, tbb[1] + tbb[3]/2)
            poly_tbb = [[tbb[0], tbb[1]], [tbb[0] + tbb[2], tbb[1]], [tbb[0] + tbb[2], tbb[1] + tbb[3]],
                        [tbb[0], tbb[1] + tbb[3]]]
            poly_tbb = np.array(poly_tbb)
            tt_pos_list.append(center)
            tt_bbox_list.append(tbb)
            tt_poly_list.append(poly_tbb)
            cpFids.append(tfid)
    #print 'compared tracking numbers %d \n' % len(tt_pos_list)

    gt_poses = np.array(gt_pos_list, dtype=float)
    tt_poses = np.array(tt_pos_list, dtype=float)

    pos_dist = np.sqrt((gt_poses[:, 0] - tt_poses[:, 0]) ** 2 + (gt_poses[:, 1] - tt_poses[:, 1]) ** 2)
    pos_precision = np.zeros((100, 1), np.float)
    for thresh in range(100):
        tracked_nums = np.sum(pos_dist <= thresh, dtype=float)
        pos_precision[thresh] = tracked_nums / len(pos_dist)

    posDistMean    = np.mean(pos_dist)
    posDistStd     = np.std(pos_dist)
    posDistRocMean = np.mean(pos_precision)

    # the X_axis is the threashold and the y_axis is the precision.
    if (bplot):
        pylab.figure()
        pylab.plot(pos_precision)
        pylab.figure()
        pylab.plot(cpFids, pos_dist)
        pylab.show(block = False)

    # # computing the rectangle overlapping
    # rect_inter_ratio = np.zeros((len(tt_bbox_list), 1), np.float)
    # for i, (tbb, gbb) in enumerate(zip(tt_bbox_list, gt_bbox_list)):
    #     rect_inter_ratio[i] = intersection_rect(tbb, gbb)
    #
    # rint_threash_range = np.arange(1, 0, -0.01).tolist()
    # rect_precision = np.zeros((len(rint_threash_range), 1), np.float)
    # for i, rint_threash in enumerate(rint_threash_range):
    #     tracked_nums = np.sum(rect_inter_ratio >= rint_threash, dtype=float)
    #     rect_precision[i] = tracked_nums / len(tt_bbox_list)
    #
    # pylab.figure()
    # pylab.plot(rint_threash_range, rect_precision)
    # pylab.figure()
    # pylab.plot(tfids, rect_inter_ratio)

    # computing the polygon region overlapping
    poly_inter_ratio = np.zeros((len(tt_poly_list), 1), np.float)
    for i, (tpoly, gpoly) in enumerate(zip(tt_poly_list, gt_poly_list)):
        poly_inter_ratio[i] = uti.intersection_poly(np.array([tpoly]), np.array([gpoly]))

    pint_threash_range = np.arange(1, 0, -0.01).tolist()
    poly_precision = np.zeros((len(pint_threash_range), 1), np.float)
    for i, pint_threash in enumerate(pint_threash_range):
        tracked_nums = np.sum(poly_inter_ratio >= pint_threash, dtype=float)
        poly_precision[i] = tracked_nums / len(tt_bbox_list)

    tbbInterGtblobMean   = np.mean(poly_inter_ratio)
    tbbInterGtblobStd    = np.std(poly_inter_ratio)
    tbbInterGtblobRocMean= np.mean(poly_precision)

    if(bplot):
        pylab.figure()
        pylab.plot(pint_threash_range, poly_precision)
        pylab.figure()
        pylab.plot(cpFids, poly_inter_ratio)
        pylab.show(block=True)

    eval_report = 'Position Distance to Gt: Mean %0.3f, Std %0.3f, posRocMean %0.3f' \
                 ' Tbb inters with Gt Blob: Mean %0.3f, Std %0.3f, iouRocMean %0.3f' \
                   % (posDistMean, posDistStd, posDistRocMean,
                      tbbInterGtblobMean, tbbInterGtblobStd, tbbInterGtblobRocMean)
    #print  eval_report
    return eval_report, pos_dist, poly_inter_ratio, pos_precision, poly_precision, cpFids

def load_zhicheng_GT():
    '''
    Set the path and GT files name, according start_id, end_id.
    :return:
    '''
    FilePath = '../ground_truth/'
    GtParam = {'ALice':{'FileName':'zhicheng20130808_22.xx_60-400_polarGT_Alice.txt',   'StartId':60, 'EndId':400},
              'Billy':{'FileName':'zhicheng20130808_22.xx_2-215_polarGT_Billy.txt',     'StartId':2,  'EndId':226},
              'Camen':{'FileName':'zhicheng20130808_22.xx_2-408_polarGT_Camen.txt',     'StartId':2,  'EndId':408},
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

def load_zhicheng_Params_TT():
    '''
    loading tracked target information in zhicheng datasets, including parameters and tracked targets tbbs and blobs.
    :return:
    '''
    FilePath = 'F:\\code\\Python\\visual_radar_tracker\\images\\zhicheng20130808_22.28\\Res_MultiKCFs_AutoParams\\'
    ParamFiles = {'ALice':'ALice_Params.txt',
                  'Billy':'Billy_Params.txt',
                  'Camen':'Camen_Params.txt',
                  'Dolphin':'Dolphin_Params.txt',
                  'Ellen':'Ellen_Params.txt'}

    TtTbbsFiles = {'ALice':'ALice_tbbs_%02d_option.txt',
               'Billy':'Billy_tbbs_%02d_option.txt',
               'Camen':'Camen_tbbs_%02d_option.txt',
               'Dolphin':'Dolphin_tbbs_%02d_option.txt',
               'Ellen':'Ellen_tbbs_%02d_option.txt'}
    TtBlobsFiles = {'ALice':'ALice_blobs_%02d_option.txt',
               'Billy':'Billy_blobs_%02d_option.txt',
               'Camen':'Camen_blobs_%02d_option.txt',
               'Dolphin':'Dolphin_blobs_%02d_option.txt',
               'Ellen':'Ellen_blobs_%02d_option.txt'}

    ParamsTT = {}
    #according the parameters' file's lines to decide howmany tbbs and blob file included.
    for key in ParamFiles:
        obj_file = FilePath + ParamFiles[key]
        with open(obj_file, 'r') as obj:
            obj_lines = obj.readlines()
        for i,line in enumerate(obj_lines):
            #tbbs or blob ttfiles's index is correlated with the line number of parameters file
            tbbsFileName = (FilePath+TtTbbsFiles[key] % i)
            blobsFileName= (FilePath+TtBlobsFiles[key] % i)
            print(tbbsFileName)
            tbbsDict   = uti.get_obj_dict(tbbsFileName)
            print (blobsFileName)
            tblobsDict = uti.get_obj_dict(blobsFileName)
            if tbbsDict == {} or tblobsDict == {}:
                print ('Empty tt file is found in %s_%d, be carefull !!!!' % (key, i))

            ptKey = '%s %d' % (key, i)
            ParamsTT.update({ptKey:{'Params':line, 'TbbsDict': tbbsDict, 'tblobsDict':tblobsDict}})
    return ParamsTT

def show_precision_MKCF_params():
    '''
    Show the multiple KCFs tracking results with different parameters.
    :return:
    '''
    GtInfo       = load_zhicheng_GT()
    ParamsTTInfo = load_zhicheng_Params_TT()

    allRes = []
    keys   = []

    for key in ParamsTTInfo:
        keys.append(key)

    keys.sort()
    for key in keys:
        gtkey = key.split(' ')[0]
        GtData= GtInfo[gtkey]['GtData']
        TbbsDict = ParamsTTInfo[key]['TbbsDict']
        BlobsDict= ParamsTTInfo[key]['tblobsDict']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(TbbsDict, GtData, bplot=False)
        print (key)
        print (ParamsTTInfo[key]['Params'])
        print (evalue_reports)
        allRes.append(key + ' ' +evalue_reports)

    for res in allRes:
        print (res)
def show_precision_KF():
    '''
    Show the tracking results with Kalman Filter, and draw the precision curve.
    :return:
    '''
    print ('\nKalman Filter\'s Tracking Results')
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_KF/'

    scores_table = np.zeros((3,5)) #scores_table, contain 3 rows, for cle, pre_cle, pre_iou
    #containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i,key in enumerate(ObjNames):
        fname = res_path + key + '_KF_Tbbs.txt'
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        #cle_fig = pylab.figure(figsize=(800/300, 600/300), dpi=300)
        #cle_ax = cle_fig.add_subplot(111)
        #pylab.xlabel('Frame', fontsize=20), pylab.ylabel('Location error(pixels)', fontsize=20)
        #cle_ax.grid(True)
        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        #cle_ax.plot(fids, cle, label=key, linewidth=2.5, alpha=1)
        #cle_ax.set_title(key)
        scores_table[0, i] = cle
        scores_table[1, i] = pre_cle
        scores_table[2, i] = pre_iou

    return scores_table

    pylab.show()

def show_precision_EOTRM():
    '''
    Show the tracking results with Extended Object Tracking based on Random Matrix, and draw the precision curve.
    :return:
    '''
    print ('\nEOTRM\'s Tracking Results')
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_EOTRM/'
    scores_table = np.zeros((3, 5))  # scores_table, contain 3 rows, for cle, pre_cle, pre_iou

    #containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i,key in enumerate(ObjNames):
        fname = res_path + key + '_EOTRM_Tbbs.txt'
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        scores_table[0, i] = cle
        scores_table[1, i] = pre_cle
        scores_table[2, i] = pre_iou

    return scores_table

def show_precision_EOTGGIW():
    '''
    Show the tracking results with Extended Object Tracking based on Random Matrix, and draw the precision curve.
    :return:
    '''
    print ('\nEOTGGIW\'s Tracking Results')
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_EOTGGIW/'
    scores_table = np.zeros((3, 5))  # scores_table, contain 3 rows, for cle, pre_cle, pre_iou

    # containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i, key in enumerate(ObjNames):
        fname = res_path + key + '_EOTGGIW_Tbbs.txt'
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        scores_table[0, i] = cle
        scores_table[1, i] = pre_cle
        scores_table[2, i] = pre_iou

    return scores_table

def show_precision_KCF():
    '''
    Show the tracking results with Extended Object Tracking based on Random Matrix, and draw the precision curve.
    :return:
    '''
    print ('\nKCF\'s Tracking Results')
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_KCF/'
    scores_table = np.zeros((3, 5))  # scores_table, contain 3 rows, for cle, pre_cle, pre_iou

    # containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i, key in enumerate(ObjNames):
        fname = res_path + key + '_KCF_Tbbs.txt'
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        scores_table[0, i] = cle
        scores_table[1, i] = pre_cle
        scores_table[2, i] = pre_iou

    return scores_table

def show_precision_ASKCF():
    '''
    Show the tracking results with Extended Object Tracking based on Random Matrix, and draw the precision curve.
    :return:
    '''
    print ('\nASKCF\'s Tracking Results')
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_ASKCF/'
    scores_table = np.zeros((3, 5))  # scores_table, contain 3 rows, for cle, pre_cle, pre_iou

    # containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i, key in enumerate(ObjNames):
        fname = res_path + key + '_ASKCF_Tbbs.txt'
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        scores_table[0, i] = cle
        scores_table[1, i] = pre_cle
        scores_table[2, i] = pre_iou

    return scores_table

def show_precision_MKL():
    '''
    Show the tracking results with Extended Object Tracking based on Random Matrix, and draw the precision curve.
    :return:
    '''
    print ('\nMKL\'s Tracking Results')
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_MKL/'
    scores_table = np.zeros((3, 5))  # scores_table, contain 3 rows, for cle, pre_cle, pre_iou

    # containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i, key in enumerate(ObjNames):
        fname = res_path + key + '_MKL_Tbbs.txt'
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        scores_table[0, i] = cle
        scores_table[1, i] = pre_cle
        scores_table[2, i] = pre_iou

    return scores_table

def show_precision_MKCF():
    '''
    Show the tracking results with Extended Object Tracking based on Random Matrix, and draw the precision curve.
    :return:
    '''
    print ('\nMKCF\'s Tracking Results')
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_MKCF/'
    scores_table = np.zeros((3, 5))  # scores_table, contain 3 rows, for cle, pre_cle, pre_iou

    # containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i, key in enumerate(ObjNames):
        fname = res_path + key + '_MKCF_Tbbs.txt'
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        scores_table[0, i] = cle
        scores_table[1, i] = pre_cle
        scores_table[2, i] = pre_iou

    return scores_table

def show_precision(tracker_name):
    '''
    show the scores of the given tracker
    :return:
    '''
    print ('\n%s\'s Tracking Results' % tracker_name)
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_%s/' % tracker_name

    # scores_table, contain 3 rows, for cle, pre_cle, pre_iou
    # the last column is for average.
    scores_table = np.zeros((3, 5))

    # containing 5 objects' tracked bounding box in KfResData
    KfResData = {}
    for i, key in enumerate(ObjNames):
        fname = res_path + key + '_%s_Tbbs.txt' % tracker_name
        ttData = uti.get_obj_dict(fname)
        KfResData.update({key:ttData})

        GtData = GtInfo[key]['GtData']
        (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
        print (key)
        print (evalue_reports)
        scores_table[0, i] = np.mean(cle)
        scores_table[1, i] = np.mean(pre_cle)
        scores_table[2, i] = np.mean(pre_iou)

    return scores_table


def draw_precision_curve():
    '''
    Draw the 5 precision results on a single plot
    :return:
    '''
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path     = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/'
    trackers = ['KF', 'EOTRM', 'EOTGGIW', 'KCF', 'ASKCF', 'MKL', 'MKCF', ]
    line_width = [1, 1.2, 1.3, 1.5, 2, 2, 2.5]
    line_style = ['o', '+', 'x', ':', '--', '-.', '-' ]
    alpha_vals = [0.4, 1, 1, 1, 1, 1, 1]

    params = {
        'axes.labelsize':  '20',
        'xtick.labelsize': '15',
        'ytick.labelsize': '15',
        'lines.linewidth':  2,
        'legend.fontsize': '16',
        #'figure.figsize': '12, 9'  # set figure size
    }
    pylab.rcParams.update(params)  # set figure parameter



    KfResData = {}

    metric_matrix = np.zeros((7, 5, 3), dtype=np.float32)

    label_font = 20
    legend_font= 14
    for j, key in enumerate(ObjNames):

        cle_fig = pylab.figure(figsize=(9,7), dpi=200)
        cle_ax = cle_fig.add_subplot(111)
        pylab.xlabel('Frame', fontsize=label_font), pylab.ylabel('Location error(pixels)', fontsize=label_font)
        cle_ax.grid(True)

        #figsize=(1200/200, 1000/200), dpi=200
        precle_fig = pylab.figure(figsize=(9,7), dpi=200)
        precle_ax = precle_fig.add_subplot(111)
        pylab.xlabel('Center location error threshold'), pylab.ylabel('Distance Precision')
        precle_ax.grid(True)

        # iou_fig = pylab.figure()
        # iou_ax = iou_fig.add_subplot(111)
        # pylab.xlabel('Frame'), pylab.ylabel('Intersection Rate')
        # iou_ax.grid(True)

        preiou_fig = pylab.figure(figsize=(9,7), dpi=200)
        preiou_ax  = preiou_fig.add_subplot(111)
        pylab.xlabel('Intersection over union threshold'), pylab.ylabel('Overlap Precision')
        preiou_ax.grid(True)

        for i, tname in enumerate(trackers):
            fname = res_path + 'Res_'+tname + '/' + key + '_'+ tname +'_Tbbs.txt'
            ttData = uti.get_obj_dict(fname)
            KfResData.update({key: ttData})

            GtData = GtInfo[key]['GtData']
            (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
            print ('%s %s tracker, mean cle %f, mean pre_cle %f, mean pre_iou %f' % (key, tname, np.mean(cle), np.mean(pre_cle), np.mean(pre_iou)))

            metric_matrix[i,j,0] = np.mean(cle)
            metric_matrix[i,j,1] = np.mean(pre_cle)
            metric_matrix[i,j,2] = np.mean(pre_iou)

            # the X_axis is the threashold and the y_axis is the precision.
            prune_cle  = cle.copy()
            prune_cle[(prune_cle>200)] = 200

            if tname == 'MKCF':
                tname = 'MKCF'+'_10'
            cle_ax.plot(fids, prune_cle, line_style[i], label=tname, linewidth = 2.5, alpha=alpha_vals[i])
            if key == 'ALice':
                titl_str = 'Alice' #change the upper case to lower case
            else:
                titl_str = key

            cle_ax.set_ylim([0, 200])
            cle_ax.set_title(titl_str, fontsize=label_font)
            cle_ax.legend(loc="upper left", fontsize=legend_font)

            #cle_ax.plot(fids, cle)

            precle_ax.plot(pre_cle, line_style[i], label= tname + '[%2.3f]'% np.mean(pre_cle), linewidth=2.5, alpha=alpha_vals[i])
            precle_ax.legend(loc="upper right", fontsize=legend_font)  # set legend location
            if key == 'ALice':
                titl_str = 'Alice' #change the upper case to lower case
            else:
                titl_str = key

            precle_ax.set_title(titl_str, fontsize=label_font)


            #iou_ax.plot(fids, IoU)
            preiou_ax.plot(np.arange(1, 0, -0.01).tolist(), pre_iou,  line_style[i],label= tname + '[%2.3f]'% np.mean(pre_iou), linewidth=2.5, alpha=alpha_vals[i])
            preiou_ax.legend(loc="upper right", fontsize=legend_font)  # set legend location
            preiou_ax.set_title(titl_str, fontsize=label_font)
            #pylab.show(block=True)

        cle_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_cle.pdf', format='pdf',dpi = 200, bbox_inches='tight')
        cle_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_cle.png', format='png', dpi=200, bbox_inches='tight')
        precle_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_precle.pdf', format='pdf',dpi = 200, bbox_inches='tight')
        precle_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_precle.png', format='png', dpi=200, bbox_inches='tight')
        #iou_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_iou.pdf', format='pdf',dpi = 300, bbox_inches='tight')
        #iou_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_iou.png', format='png', dpi=300, bbox_inches='tight')
        preiou_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_preiou.pdf', format='pdf',dpi = 200, bbox_inches='tight')
        preiou_fig.savefig('/Users/yizhou/code/MKCF_MAC_Python3.6/results/'+titl_str+'_preiou.png', format='png', dpi=200, bbox_inches='tight')

        #pylab.show(block=True)
        cle_fig.clf()
        precle_fig.clf()
        #iou_fig.close()
        preiou_fig.clf()
        #pylab.show()
    #print(metric_matrix)


    for m, tname in enumerate(trackers):
        mcle = metric_matrix[m, :, 0]
        print ('%s\t average mean of cle    in five objects %3.3f'    % (tname, np.mean(mcle)))
        #print (mcle)
    for m, tname in enumerate(trackers):
        mprecle = metric_matrix[m, :, 1]
        print ('%s\t average mean of precle in five objects %3.3f' % (tname, np.mean(mprecle)))
        #print (mprecle)
    for m, tname in enumerate(trackers):
        mpreiou = metric_matrix[m, :, 2]
        print ('%s\t average mean of preiou in five objects %3.3f' % (tname, np.mean(mpreiou)))
        #print (mpreiou)

    print('cle of 7 trackers\n')
    print(metric_matrix[:,:,0])

    print('precle of 7 trackers\n')
    print(metric_matrix[:,:,1])

    print('preiou of 7 trackers\n')
    print(metric_matrix[:,:,2])

    print ('finish painting')


#compare different numbers of MKCF, m=3,5,8,10,15
def draw_precision_curve_MKCF():
    '''
    Draw the 5 precision results on a single plot
    :return:
    '''
    ObjNames = []
    GtInfo = load_zhicheng_GT()
    for key in GtInfo:
        ObjNames.append(key)
    ObjNames.sort()

    res_path     = 'F:\\code\\Python\\visual_radar_tracker\\images\\zhicheng20130808_22.28\\Res_MKCF\\'
    trackers     = ['3','5','8','10','15']
    line_width = [1, 1.2, 1.5, 2, 2.5]
    line_style = ['o', '+', '--', '-.', '-' ]
    alpha_vals = [0.3, 1, 1, 1, 1]

    cle_fig = pylab.figure()
    cle_ax    = cle_fig.add_subplot(111)
    pylab.xlabel('Frame ids', fontsize=15), pylab.ylabel('Location error(pixels)', fontsize=15)

    precle_fig = pylab.figure()
    precle_ax  = precle_fig.add_subplot(111)
    pylab.xlabel('Center Location error threshold', fontsize=15), pylab.ylabel('Precision', fontsize=15)

    iou_fig   = pylab.figure()
    iou_ax    = iou_fig.add_subplot(111)
    pylab.xlabel('Frame ids'), pylab.ylabel('Intersection Rate')

    preiou_fig = pylab.figure()
    preiou_ax  = preiou_fig.add_subplot(111)
    pylab.xlabel('Intersection over union threshold'), pylab.ylabel('Precision')

    metric_matrix = np.zeros((5,5,3), dtype=np.float32)

    KfResData = {}
    for j, key in enumerate(ObjNames):
        for i, tname in enumerate(trackers):
            fname = res_path + '20180323_votepsr=10\\20180323_M='+ tname +'\\'+key+'_MKCF_Tbbs.txt'
            ttData = uti.get_obj_dict(fname)
            KfResData.update({key: ttData})

            GtData = GtInfo[key]['GtData']
            (evalue_reports, cle, IoU, pre_cle, pre_iou, fids) = show_tbb_precision(ttData, GtData, bplot=False)
            # the X_axis is the threashold and the y_axis is the precision.

            prune_cle  = cle.copy()
            prune_cle[(prune_cle>200)] = 200
            cle_ax.plot(fids, prune_cle)
            precle_ax.plot(pre_cle, line_style[i], label= tname + '[%2.3f]'% np.mean(pre_cle), linewidth=2.5, alpha=alpha_vals[i])
            precle_ax.legend(loc="lower right", fontsize=15)  # set legend location
            if key == 'ALice':
                titl_str = 'Alice' #change the upper case to lower case
            else:
                titl_str = key

            precle_ax.set_title(titl_str)


            iou_ax.plot(fids, IoU)
            preiou_ax.plot(np.arange(1, 0, -0.01).tolist(), pre_iou,  line_style[i],label= tname + '[%2.3f]'% np.mean(pre_iou), linewidth=2.5, alpha=alpha_vals[i])
            preiou_ax.legend(loc="upper right", fontsize=15)  # set legend location
            preiou_ax.set_title(titl_str)
            pylab.show(block=False)

            metric_matrix[i,j,0] = np.mean(cle)
            metric_matrix[i,j,1] = np.mean(pre_cle)
            metric_matrix[i,j,2] = np.mean(pre_iou)
            print ('%s number of trackers %s of KMCF, mean cle %3.3f, mean pre_cle % 3.3f, mean pre_iou % 3.3f' % (key, tname, np.mean(cle), np.mean(pre_cle), np.mean(pre_iou)))

        # cle_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_cle.pdf', format='pdf',dpi = 1200, bbox_inches='tight')
        # cle_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_cle.png', format='png', dpi=1200, bbox_inches='tight')
        # precle_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_precle.pdf', format='pdf',dpi = 1200, bbox_inches='tight')
        # precle_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_precle.png', format='png', dpi=1200, bbox_inches='tight')
        # iou_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_iou.pdf', format='pdf',dpi = 1200, bbox_inches='tight')
        # iou_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_iou.png', format='png', dpi=1200, bbox_inches='tight')
        # preiou_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_preiou.pdf', format='pdf',dpi = 1200, bbox_inches='tight')
        # preiou_fig.savefig('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\exp\\'+titl_str+'_preiou.png', format='png', dpi=1200, bbox_inches='tight')

        cle_ax.clear()
        precle_ax.clear()
        iou_ax.clear()
        preiou_ax.clear()


    for m, tname in enumerate(trackers):
        mcle = metric_matrix[m,:,0]
        print ('%s average mean of cle in five objects %f' %(tname, np.mean(mcle)))
        print (mcle)

        mprecle = metric_matrix[m, :, 1]
        print ('%s average mean of precle in five objects %f' % (tname, np.mean(mprecle)))
        print (mprecle)

        mpreiou = metric_matrix[m, :, 2]
        print ('%s average mean of preiou in five objects %f' % (tname, np.mean(mpreiou)))
        print (mpreiou)
    #pylab.show()
    print ('finish painting')

def format_time_table():
    """
    format the time table for all the trackers.
    :return:
    """
    tracker_names=['KF', 'EOTRM', 'EOTGGIW', 'KCF', 'ASKCF', 'MKL', 'MKCF_3', 'MKCF_10', 'MKCF' ]
    target_names =['ALice', 'Billy', 'Camen', 'Dolphin', 'Ellen']
    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/'

    #from numpy import np

    time_table = np.zeros((len(tracker_names), len(target_names)+1))
    table_title = ''

    for target in tracker_names:
        table_title +=  '%s\t' % target

    table_title += 'Average'

    dash = '-' * 80
    print('Time consuming table')
    print('{:<10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:>10s}'.format('Tracker',target_names[0], target_names[1], target_names[2], \
                                                              target_names[3], target_names[4], 'Average'))
    print(dash)
    for j,tname in enumerate(tracker_names):
        # if tname == 'KCF': #or tname.__contains__('MKCF_'):
        #       continue
        time_file_name = res_path + 'Res_' + tname + '/' + 'cost_time.txt'
        tfile = open(time_file_name, 'r')

        for i in range(len(target_names)+1):
            line = tfile.readline()
            [target, _,  time_cost, _] = line.split(' ')
            time_table[j, i] = float(time_cost) * 1000

        print('{:<10s}{:>10.1f}{:>10.1f}{:>10.1f}{:>10.1f}{:>10.1f}{:>10.1f}'.format(tname, time_table[j,0], time_table[j,1], \
                                                                               time_table[j,2], time_table[j,3], time_table[j,4], time_table[j,5]))
def format_scores_table(scores_table, table_name, datatype ='float'):
    '''
    print score tables for the paper
    :param scores_table:
    :param name of the _table:
    :return:
    '''
    tracker_names=['KF', 'EOTRM', 'EOTGGIW', 'KCF', 'ASKCF', 'MKL', 'MKCF_3', 'MKCF_10', 'MKCF' ]
    target_names =['ALice', 'Billy', 'Camen', 'Dolphin', 'Ellen']



    dash = '-' * 80
    print(table_name)
    print('{:<10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}'.format('Tracker',target_names[0], target_names[1], target_names[2], \
                                                              target_names[3], target_names[4], 'Average'))
    print(dash)
    for j,tname in enumerate(tracker_names):
        if datatype == 'int':
            print('{:<10s}&{:>10.0f}&{:>10.0f}&{:>10.0f}&{:>10.0f}&{:>10.0f}&{:>10.0f}'.format(tname, scores_table[j, 0],scores_table[j, 1], \
                                                                                         scores_table[j, 2],       scores_table[j, 3], \
                                                                                         scores_table[j, 4], np.mean(scores_table[j, :])))
        if datatype == 'float':
            print('{:<10s}&{:>10.3f}&{:>10.3f}&{:>10.3f}&{:>10.3f}&{:>10.3f}&{:>10.3f}'.format(tname, scores_table[j, 0], scores_table[j, 1], \
                                                                                     scores_table[j, 2], scores_table[j, 3],        \
                                                                                     scores_table[j, 4], np.mean(scores_table[j, :])))
def plot_trackerno_vs_dp_op():

    trackno = [3,5,8,10,15]
    dps     = [0.595, 0.746, 0.780, 0.923, 0.912]
    ops     = [0.359, 0.441, 0.469, 0.557, 0.544]

    fig = pylab.figure()
    ax    = fig.add_subplot(111)

    ax.plot(trackno, dps, 'bo-', label='distance precision', color='blue',linewidth=2.5, markersize=10)
    ax.plot(trackno, ops, 'gv-', label='overlap  precision', color='green',linewidth=2.5,markersize=10)
    ax.legend(loc="upper left", fontsize=15)  # set legend location
    pylab.xlabel('Number of trackers', fontsize=15), pylab.ylabel('Precision', fontsize=15)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18])
    pylab.show(block=True)
    #pylab.set_title()

def plot_highlight():
    img = cv2.imread('F:\\code\\Python\\visual_radar_tracker\\images\\paper\\5TrackerRes\\ALice\\0086.jpg')
    pylab.figure()
    pylab.imshow(img)
    pylab.show(block= True)

if __name__ == '__main__':
    pylab.ioff()
    # format_time_table()
    #
    #
    # cle_table = np.zeros((9,5))
    # precle_table = np.zeros((9,5))
    # preiou_table = np.zeros((9, 5))
    # tracker_names = ['KF', 'EOTRM', 'EOTGGIW', 'KCF', 'ASKCF', 'MKL', 'MKCF_3', 'MKCF_10', 'MKCF']
    # for i,tname in enumerate(tracker_names):
    #     # if tname.__contains__('MKCF_'):
    #     #     continue
    #     scores_table = show_precision(tname)
    #     cle_table[i, :] = scores_table[0, :] # zero row of scores_table for cle of 5 target
    #     precle_table[i,:] = scores_table[1, :]
    #     preiou_table[i,:] = scores_table[2, :]
    #
    # format_scores_table(cle_table,    'Mean of center location error',  'int')
    # format_scores_table(precle_table, 'Mean of distance precision', 'float')
    # format_scores_table(preiou_table, 'Mean of overlap precision', 'float')
    #
    # print('done')
    # show_precision_KF()
    # show_precision_EOTRM()
    # show_precision_EOTGGIW()
    # show_precision_KCF()
    # show_precision_ASKCF()
    # show_precision_MKL()
    # show_precision_MKCF()
    #pylab.show()


    #draw_precision_curve_MKCF()
    draw_precision_curve()
    #plot_trackerno_vs_dp_op()

    #plot_highlight()