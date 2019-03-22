#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of video maker,
# Choosing a directory, compress all the images into a video.


#OpenCV Video Writer on Mac OS X
#https://gist.github.com/takuma7/44f9ecb028ff00e2132e

from cv2 import cv2
import numpy as np
import os
import utility as uti

def view_trackers():
    tracker_names = ['KF', 'EOTRM', 'EOTGGIW', 'KCF', 'ASKCF', 'MKL', 'MKCF']
    #colors[red, yellow, light blue, white, mud green, violite, light green]
    tracker_colors= [(0,0,255),(0,255,255), (239,200,148), (255,255,255), (24,115,84), (227,5,198), (97,241,115) ]
    target_names = ['ALice', 'Billy', 'Camen', 'Dolphin', 'Ellen']
    res_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/'
    ResData = {}

    #loading target's bounding box '*_tbbs.txt' file for 5 targets in 7 trackers.
    for tracker_name in tracker_names:
        TrackerResData = {}
        for target_name in target_names:
            ttData = {}
            fname = res_path + 'Res_' + tracker_name + '/' + target_name + '_' + tracker_name + '_Tbbs.txt'
            ttData = uti.get_obj_dict(fname)
            TrackerResData.update({target_name: ttData})
        ResData.update({tracker_name: TrackerResData})
    print('size of the results data in bytes: %d'% ResData.__sizeof__())

    image_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/sequences/gray/'
    video_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/video/'
    # video = cv2.VideoWriter(video_path + '7trackers_5targets.mp4', cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 30,
    #                     (2048, 600))

    video = cv2.VideoWriter(video_path + '7trackers_5targets.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                            (2048, 600))
    img_files = os.listdir(image_path)
    img_files.sort()
    for file in img_files:
        img_file = os.path.join(image_path, file)
        if img_file.endswith('.png'):
            img = cv2.imread(img_file)

            id = int(file.split('.')[0])
            if id > 409:
                break
            frame_id = 'frame %d' % id
            for i,tracker_name in enumerate(tracker_names):
                cv2.putText(img, frame_id, org=(20, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(245,255,118),#show the tracker's color in rectangle and label.
                            thickness=2, lineType=cv2.LINE_AA)
                img = cv2.rectangle(img, (30, 60+i*40), (80, 90+i*40), tracker_colors[i], 2)
                cv2.putText(img, tracker_name, org=(80, 90+i*40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=tracker_colors[i],
                            thickness=2, lineType=cv2.LINE_AA)

                for target_name in target_names:
                    if frame_id in  ResData[tracker_name][target_name]:
                        tbb = ResData[tracker_name][target_name][frame_id]['BoundingBox']
                        if len(tbb)>0:
                            tp  = (tbb[0], tbb[1])
                            br  = (int(tbb[0]+tbb[2]), int(tbb[1]+tbb[3]))
                            img = cv2.rectangle(img, tp, br, tracker_colors[i], 2)
                            if(tracker_name == 'MKCF'):
                                cv2.putText(img, target_name, org=tp,
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(245,255,118),
                                            thickness=2, lineType=cv2.LINE_AA)

            video.write(img)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    view_trackers()
    exit(0)
    image_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/Res_MKCF/ALice'
    video_path = '/Users/yizhou/code/MKCF_MAC_Python3.6/results/video/'

    #video = cv2.VideoWriter('inner_river_alice.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (2048,600))
    #video = cv2.VideoWriter(video_path+'inner_river_alice.avi', cv2.VideoWriter_fourcc('M', 'P', '4', 'S'), 10, (2048, 600))
    video = cv2.VideoWriter(video_path + 'inner_river_alice.mp4', cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 10, (2048, 600))
    #video = cv2.VideoWriter('inner_river_alice.avi', -1)
    img_files = os.listdir(image_path)
    img_files.sort()
    for file in img_files:
        img_file = os.path.join(image_path, file)
        if img_file.endswith('.png'):
            img = cv2.imread(img_file)

            img = cv2.rectangle(img, (30, 60), (80, 90), (0, 255, 255), 2)
            cv2.putText(img, 'rectangle of the component KCF, whose PSR>=10', org=(80, 90),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)

            img = cv2.rectangle(img, (30, 100), (80, 130), (125, 125, 125), 2)
            cv2.putText(img, 'rectangle of the component KCF, whose PSR<10', org=(80, 130),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(125, 125, 125),
                        thickness=2, lineType=cv2.LINE_AA)

            img = cv2.rectangle(img, (30, 140), (80, 170), (0,0,255), 2)
            cv2.putText(img, 'fused rectangle', org=(80, 170),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2, lineType=cv2.LINE_AA)

            img = cv2.rectangle(img, (30, 180), (80, 210), (255, 0, 0), 2)
            cv2.putText(img, 'segmented  rectangle', org=(80, 210),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                        thickness=2, lineType=cv2.LINE_AA)
            video.write(img)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
    video.release()
    cv2.destroyAllWindows()