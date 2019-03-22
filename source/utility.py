#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
#cv2.ocl.setUseOpenCL(False)
import numpy as np
import ast

# import matplotlib
# matplotlib.use('wxAgg')
import copy

import time, ctypes
class msTimer(object):
    def __init__(self):
        self.cll = ctypes.c_longlong(0)
        #computing cpu frequency
        ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(self.cll))
        self.freq = self.cll.value
    def start(self):
        #mark start cpu counter
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(self.cll))
        self.st = self.cll.value
        return self.st
    def count(self):
        #mark end cpu counter
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(self.cll))
        self.end = self.cll.value
        timer    = (self.end - self.st)/(self.freq+0.)
        return timer


def frame_normalize(frame):
    '''
    :param frame: source frame
    :return: normalized frame in double float form
    '''
    # 归一化操作
    # dframe is for computing in float32
    dframe = frame.astype(np.float32) #for the convenient of opencv operation
    #normalization
    dframe = (dframe - dframe.min()) / (dframe.max() - dframe.min())
    return dframe



def get_obj_dict(obj_file):
    #TT_file is for tracked_target file
    #GT_file is for ground_truth file

    with open(obj_file, 'r') as obj:
        obj_lines = obj.readlines()

    obj_dict = {}

    for obj_line in obj_lines:
        one_frame = ast.literal_eval(obj_line)
        obj_dict.update(one_frame)
    return obj_dict

def get_env_roi_mask(roi_poly_name, height=600, width = 2048):
    '''
    get envioronment region of interest mask
    :return:
    '''
    ROI_dict = get_obj_dict(roi_poly_name)
    canvas = np.zeros((height, width), np.float)


    for key in ROI_dict:
        roi = ROI_dict[key]
        poly = roi['Polygon']

        #cv2.fillConvexPoly(canvas, np.array(poly), [255]*3, 8)
        cv2.fillPoly(canvas, np.array([poly]), 1, 8)

    mask = (canvas ==  1)*1.0
    mask = mask.astype(np.float32)
    #cv2.imshow('mask', canvas)
    #cv2.waitKey()
    return mask

def get_Inter_ROC(InterRatios):
    InterRatios = np.array(InterRatios)
    threash_range = np.arange(1,0,-0.01).tolist()
    inter_precision = np.zeros((len(threash_range), 1), np.float)
    for i,int_threash in enumerate(threash_range):
        tracked_nums = np.sum(InterRatios >= int_threash, dtype=float)
        inter_precision[i] = tracked_nums / len(InterRatios)
    RocArea = np.sum(inter_precision)
    # pylab.figure()
    # pylab.plot(threash_range, inter_precision)
    return RocArea

def intersection_rect(recta, rectb):
    '''
    Intersection area of two rectangles.
    :param recta:
    :param rectb:
    :return: iou rate.
    '''
    tlx = max(recta[0], rectb[0])
    tly = max(recta[1], rectb[1])
    brx = min(recta[0]+recta[2], rectb[0]+rectb[2])
    bry = min(recta[1]+recta[3], rectb[1]+rectb[3])

    intersect_area = max(0., brx-tlx+1) * max(0., bry-tly+1)
    iou = intersect_area/(recta[2]*recta[3] + rectb[2]*rectb[3] - intersect_area + np.spacing(1))
    return iou

def is_rect_contained(rect_small, rect_big):
    '''
    Judge a rect_small is contained or verse vice
    No matter which rect is contained , return true
    :param rect_small:
    :param rect_big:
    :return:
    '''
    tlx = max(rect_small[0], rect_big[0])
    tly = max(rect_small[1], rect_big[1])
    brx = min(rect_small[0]+rect_small[2], rect_big[0]+rect_big[2])
    bry = min(rect_small[1]+rect_small[3], rect_big[1]+rect_big[3])

    intersect_area = max(0., brx-tlx+1) * max(0., bry-tly+1)
    small_area = rect_small[2]*rect_small[3]
    big_area   = rect_big[2]*rect_big[3]

    is_contained = False
    if intersect_area*1./small_area  > 0.75:
        is_contained = True
    if intersect_area*1./big_area > 0.75:
        is_contained = True
    return is_contained

def intersection_poly(poly1, poly2):

    #finding the minimum containing window for polyfill
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if (rect1[2]*rect1[3]<=0) or (rect2[2]*rect2[3]<=0):
        print ('wrong bounding box for polygon in intersection_poly function\n')
    tlx = min(rect1[0], rect2[0])
    tly = min(rect1[1], rect2[1])
    brx = max(rect1[0]+rect1[2], rect2[0]+rect2[2])
    bry = max(rect1[1]+rect1[3], rect2[1]+rect2[3])

    width = brx - tlx + 1
    height= bry - tly + 1

    #generating canvas img for the maximum bounding box for two polygon
    img1 = np.zeros((height, width), np.uint8)
    img2 = np.copy(img1)

    offset1 = np.tile([[[tlx, tly]]], (len(poly1), 1, 1))
    offset2 = np.tile([[[tlx, tly]]], (len(poly2), 1, 1))
    poly1 = poly1 - offset1
    poly2 = poly2 - offset2

    #Fill polygon.
    #img1 = cv2.fillPoly(img1, poly1, [255],8)
    img1 = cv2.fillConvexPoly(img1, poly1, [255],8)
    img2 = cv2.fillConvexPoly(img2, poly2, [255],8)
    #computing the pixels for overlapped region.
    mask1 = (img1>0)
    mask2 = (img2>0)
    intersection = np.sum(mask1*mask2)
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)

    inter_ratio = intersection/(area1+area2-intersection+np.spacing(1))

    ## for test monitoring
    # intgray = img1 * img2 * 255
    # canvas = cv2.cvtColor(intgray, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(canvas, [poly1], 0, (255, 0, 0), 2)
    # cv2.drawContours(canvas, [poly2], 0, (0, 255, 0), 2)
    # cv2.imshow('inter', canvas)

    return inter_ratio

def draw_rect(img, rect, color = (0,255,0), thick = 1):
    '''
    draw a rectangle on img, for convinient using than cv2
    :param img:
    :param rect:
    :return: img
    '''
    p1 = (rect[0], rect[1])
    p2 = (rect[0]+rect[2], rect[1]+rect[3])
    img = cv2.rectangle(img, p1, p2, color, thick)
    return img

def draw_blob_geo(blob_geo, img):
    if blob_geo != {}:
        #draw metric elements
        x, y, w, h = blob_geo['BoundingBox']
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        rotatedBox = blob_geo['RotatedBox']
        box = cv2.boxPoints(rotatedBox)
        img = cv2.ellipse(img, rotatedBox, (0, 255, 0), 3)
        # box = np.int0(box)
        # img = cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
        cnt = np.array([blob_geo['Polygon']])
        cv2.polylines(img, cnt, True,  (255, 0, 0), 2)
        # cv2.drawContours(img, cnt, -1, (255, 0, 0), 2)
    return img

def get_imageROI(im, rect):

    # ys = pylab.floor(rect[1])  + pylab.arange(rect[3], dtype=int)
    # xs = pylab.floor(rect[0])  + pylab.arange(rect[2], dtype=int)
    ys = np.floor(rect[1])  + np.arange(rect[3], dtype=int)
    xs = np.floor(rect[0])  + np.arange(rect[2], dtype=int)
    ys = ys.astype(int)
    xs = xs.astype(int)

    # check for out-of-bounds coordinates,
    # and set them to the values at the borders
    ys[ys < 0] = 0
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    xs[xs < 0] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1
    #roi = im[pylab.ix_(ys, xs)]
    roi = im[np.ix_(ys, xs)]
    return roi

def blob_geometry(blob, tlx=0, tly=0):
    if cv2.__version__.startswith("3."):
        (_, contours, _) = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        npts = len(cnt)
        if (npts == 0):
            print('!!!blob find no countours!!!\n')
            return {}

        offset_array = np.tile([[[tlx, tly]]], (npts, 1, 1))

        cnt = cnt + offset_array

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx_contour = cv2.approxPolyDP(cnt, epsilon, True)

        blob_geo = {}
        if len(approx_contour) > 2:
            moment = cv2.moments(approx_contour)
            cx = int(moment["m10"] / (moment["m00"] + np.spacing(1)))
            cy = int(moment["m01"] / (moment["m00"] + np.spacing(1)))
            rect = cv2.boundingRect(approx_contour)
            rotatedBox = cv2.minAreaRect(approx_contour)
            centroid = (cx, cy)
            boundingBox = rect
            center = (int(rect[0] + rect[2] / 2), int(rect[1] + rect[3] / 2))

            polypts = cnt.reshape((cnt.shape[0], 2))
            polypts = polypts.tolist()
            blob_geo = {'Polygon': polypts, 'Centroid': centroid, 'Center': center,
                        'BoundingBox': boundingBox, 'RotatedBox': rotatedBox}
    if cv2.__version__.startswith("4."):
        (contours, _) = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_list = []

        for contor in contours:
            npts = len(contor)
            if npts == 0:
                print('!!!blob find no countours!!!\n')
                return {}
            # add image coordinates offset for the contour in ROI
            offset_array = np.tile([[[int(tlx), int(tly)]]], (npts, 1, 1))
            contor = contor + offset_array

            perimeter = cv2.arcLength(contor, True)
            epsilon = 0.01 * cv2.arcLength(contor, True)
            approx_contour = cv2.approxPolyDP(contor, epsilon, True)
            blob_geo = {}
            if len(approx_contour) > 2:
                moment = cv2.moments(approx_contour)
                cx = int(moment["m10"] / (moment["m00"] + np.spacing(1)))
                cy = int(moment["m01"] / (moment["m00"] + np.spacing(1)))
                rect = cv2.boundingRect(approx_contour)
                rotated_box = cv2.minAreaRect(approx_contour)
                centroid = (cx, cy)
                bounding_box = rect
                center = (int(rect[0] + rect[2] / 2), int(rect[1] + rect[3] / 2))
                poly_pts = contor.reshape((contor.shape[0], 2))
                poly_pts = poly_pts.tolist()
                blob_geo = {'Polygon': poly_pts, 'Centroid': centroid, 'Center': center,
                            'BoundingBox': bounding_box, 'RotatedBox': rotated_box}
                blob_list.append(blob_geo)

    return blob_list

def blob_seg(frame, bbox, morph_iter = 2, dilate_iter = 3, dist_threash = 0.2, sub_window=(128,64), min_area = 64):
    '''
    Exteneding bbox to sub_window of a frame.
    Find all the blob information in the sub_window roi.
    :param frame:       source image for threashold and connected component computing
    :param bbox:        object's bounding box, in rect format
    :param sub_window:  extended roi window size for an object
    :param morph_iter, dilate_iter: is for morphologyEx
    :param dist_threash: for dist_threash, first vesion using 0.6, now changing to 0.2
    :param min_area:    minimum blob's size should bigger than min_area (default is 8*8)
    :return:            blob_list (each element is a blob_info_dict, dict has 'Polygon', 'BoundingBox', 'RotatedBox', etc.)
    '''
    pos = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    posx, posy = pos

    subw,subh = sub_window
    tlx = posx - subw / 2
    tly = posy - subh / 2
    # get subwindow and threash on the WHOLE image's mean.
    frame_roi = get_imageROI(frame, [tlx, tly, subw, subh])

    # frame_roi_mask = (frame_roi > np.mean(frame))
    # 自动门限仅仅对灰度图像可用。
    frame_roi = frame_roi*255
    frame_roi = frame_roi.astype(np.uint8)
    # ret, frame_roi_mask = cv2.threshold(frame_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, threash = cv2.threshold(frame_roi, 0, 255, cv2.THRESH_OTSU)
    frame_roi_mask = frame_normalize(threash)

    # frame_roi_mask = (frame_roi > echo_threash)
    # threash = frame_roi_mask * 255
    # threash = thresh.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(threash, cv2.MORPH_OPEN, kernel, iterations=morph_iter)


    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iter)
    #cv2.imshow('sure_bg', sure_bg)


    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, dist_threash * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    #cv2.imshow('fg', sure_fg)
    #cv2.imshow('openning', opening)
    #cv2.waitKey(1)

    # Marker labelling
    ret, markers = cv2.connectedComponents(opening)
    #ret, markers = cv2.connectedComponents(sure_fg)
    # pylab.figure()
    # pylab.imshow(markers)


    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # pylab.figure()
    # pylab.imshow(markers)
    # #pylab.show()


    img = frame_normalize(frame_roi)*255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img, markers)

    # pylab.figure()
    # pylab.imshow(markers)
    # pylab.show(block=True)
    # cv2.imshow('sure_fg', sure_fg)
    # cv2.waitKey(1)

    #ret, markers = cv2.connectedComponents(opening)

    inter_ratio_max = 0.
    blob_list = []
    for id in range(markers.max()+1):
        #only chosing the foreground's markers
        #mask = ((markers == id)*frame_roi_mask)

        #only take the sure_foreground for blob geometry computing.
        mask = ((markers == id) *sure_fg )
        if(np.sum(mask) > min_area):
            #candi_blob = mask * 255
            candi_blob = (markers == id)*255
            candi_blob = candi_blob.astype(np.uint8)
            # ## Draw contours testing
            # (contours, _) = cv2.findContours(candi_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            #candi_blob_info = blob_geometry(candi_blob, tlx, tly)
            #if candi_blob_info != {}:
            #blob_list.append(candi_blob_info)
            blobs = blob_geometry(candi_blob, tlx, tly)
            blob_list.extend(blobs)
     # frame_canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
     # for blob in blob_list:
     #     draw_blob_geo(blob, frame_canvas)
     # frame_canvas = draw_rect(frame_canvas, (posx-subw/2, posy-subh/2, subw, subh), (255,255,255))
     # cv2.imshow('blob_seg', frame_canvas)
     # cv2.waitKey(1)

    #Testing the subwindow's segmentation effect
    sub_blob_list = []
    for id in range(markers.max()+1):
        # only chosing the foreground's markers
        # mask = ((markers == id)*frame_roi_mask)
        mask = ((markers == id) * sure_fg)
        if np.sum(mask) > min_area:
            #sub_blob = mask * 255
            sub_blob = (markers == id)*255
            sub_blob = sub_blob.astype(np.uint8)
            sub_blobs = blob_geometry(sub_blob)
            #sub_blob_info = blob_geometry(sub_blob)
            #if sub_blob_info != {}:
            #sub_blob_list.append(sub_blob_info)
            sub_blob_list.extend(sub_blobs)
    sub_frame_canvas = cv2.cvtColor(frame_roi,cv2.COLOR_GRAY2BGR)
    sub_frame_canvas = frame_normalize(sub_frame_canvas)*255
    sub_frame_canvas = sub_frame_canvas.astype(np.uint8)
    for i,sbb in enumerate(sub_blob_list):
        draw_blob_geo(sbb, sub_frame_canvas)
        rect = sbb['BoundingBox']
        tlpt = (rect[0], rect[1])
        cv2.putText(sub_frame_canvas, str(i), org=tlpt,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 0),
                    thickness=1, lineType=cv2.LINE_AA)
    #cv2.imshow('sub_blob', sub_frame_canvas)
    #pylab.imshow(sub_frame_canvas, animated=True)
    return blob_list, sub_frame_canvas, sub_blob_list

def vote_blob(blob_list, tbox_list):
    '''
    Computing all the bob and tbox's overlapping area, and intersection_ratio,
    Each tracker's tbox vote for a candidate blob.
    The top voted  blob is choosing as the target's blob
    :param blob_list: contains all the blob, each element has a dict('Polygon', 'BoundingBox','Center',.etc), sa blog_geometry
    :param tbox_list: contains all the tracker's estimated bounding box for the target
    :return: (blob_id, blob_score)the blob best matching all the tracker's bbox, measured by overlapped area ratio
    '''

    #Note for empty blob_list or tbox_list
    blen = len(blob_list)
    tlen = len(tbox_list)
    if blen*tlen == 0:
        #return null blob and 0 blob_score
        return {},0,0


    scores = np.zeros((tlen, blen), np.float)
    votes = np.zeros((tlen, blen), np.uint)

    #computing each blob and tbox's overlapping ratio, get scores_matrix
    #scores_matrix each row means one tracker in different blob's overlapped ratio
    #each col means one blob in different tracker's bbox's overlapped ratio
    #       scores_matrix(3trackers and 3 blobs)                vote_matrix
    #           t\b |  b1 | b2 | b3                      t\b | b1 | b2 | b3
    #           t1  | 0.3 |0.2 | 0.1                      t1 | 1  | 0  | 0
    #           t2  | 0.1 |0.1 | 0.2                      t2 | 0  | 0  | 1
    #           t3  | 0.4 |0.3 | 0.1                      t3 | 1  | 0  | 0
    #                                                          2    0    1 (2 trackers vote blob1, 1 tracker vote blob3)
    for i,tbb in enumerate(tbox_list):
        for j,blob in enumerate(blob_list):
            poly_tbb = [[tbb[0], tbb[1]], [tbb[0]+tbb[2], tbb[1]], [tbb[0]+tbb[2], tbb[1]+tbb[3]], [tbb[0], tbb[1]+tbb[3]]]
            poly_tbb = np.array([poly_tbb])
            poly_blob= np.array([blob['Polygon']])
            blob_bb  = blob['BoundingBox']
            #scores[i,j] = intersection_poly(poly_tbb, poly_blob)
            scores[i, j] = intersection_rect(tbb, blob_bb)

    #using the maximum in each socre_matrix row, to vote the blob
    for row in range(tlen):
        row_score = scores[row,:]
        row_vote  = votes[row,:]
        #selects the blob has overlapped pixels
        if row_score.max() > 0:
            row_vote[row_score==row_score.max()] = 1
        votes[row,:] = row_vote

    #vertical suming for counting votes
    #counts = np.sum(votes, axis = 0)

    counts = np.mean(scores, axis = 0)
    blob_id = np.argmax(counts)
    blob = blob_list[blob_id]
    blob_score = np.mean(scores[:, blob_id])


    #print 'voted blob id %d\n' % blob_id
    #print scores
    return blob, blob_score, blob_id


def show_precision(tt_dict, gt_dict):
    import pylab
    '''
    center point's distance and tracked bounding box(tbb) overlapped with ground truth polygon
    tbb overlapped gt_polygon, is similar to the blob_score, which is tbb overlapped with seg_blob' polygon
    :param tt_dict:
    :param gt_dict:
    :return:
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
            gt_bbox_list.append(gt_info_dict['BoundingBox'])
            gt_poly_list.append(gt_info_dict['Polygon'])


            tbb = tt_info_dict['BoundingBox']
            center = (tbb[0] + tbb[2]/2, tbb[1] + tbb[3]/2)
            poly_tbb = [[tbb[0], tbb[1]], [tbb[0] + tbb[2], tbb[1]], [tbb[0] + tbb[2], tbb[1] + tbb[3]],
                        [tbb[0], tbb[1] + tbb[3]]]

            tt_pos_list.append(center)
            tt_bbox_list.append(tbb)
            tt_poly_list.append(poly_tbb)
    print ('compared tracking numbers %d \n' % len(tt_pos_list))

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
        poly_inter_ratio[i] = intersection_poly(np.array([tpoly]), np.array([gpoly]))

    pint_threash_range = np.arange(1, 0, -0.01).tolist()
    poly_precision = np.zeros((len(pint_threash_range), 1), np.float)
    for i, pint_threash in enumerate(pint_threash_range):
        tracked_nums = np.sum(poly_inter_ratio >= pint_threash, dtype=float)
        poly_precision[i] = tracked_nums / len(tt_bbox_list)

    pylab.figure()
    pylab.plot(pint_threash_range, poly_precision)
    pylab.figure()
    pylab.plot(tfids, poly_inter_ratio)
    pylab.show()
