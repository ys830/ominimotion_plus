import os
import cv2
import csv
import numpy as np
import time
import peakutils
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics


def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=False):
    frame_numbers = []
    keyframe_indices = []
    keyframePath = dest+'/keyFrames'
    csvPath = dest+'/txtFile'
    # path2file = csvPath + '/output.txt'
    prepare_dirs(keyframePath, csvPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()
    
    # Read until video is completed
    for i in range(length):
        ret, frame = cap.read()
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        frame_numbers.append(int(frame_number))

        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time-Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

    cap.release()
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y-base, Thres, min_dist=1)
    

    cnt = 1
    for x in indices:
        keyframe_indices.append(frame_numbers[x])
        # cv2.imwrite(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'), full_color[x])
        cv2.imwrite(os.path.join(keyframePath , str(frame_numbers[x]).zfill(5) +'.jpg'), full_color[x])
        cnt +=1
        # log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        log_message = 'keyframe ' + str(cnt) + ' happened at frame ' + str(frame_numbers[x]) + '.'
        if(verbose):
            print(log_message)

    with open(os.path.join(csvPath, 'frame_numbers.txt'), 'w') as frame_file:
        for frame_number in keyframe_indices:
            frame_file.write(str(frame_number) + '\n')