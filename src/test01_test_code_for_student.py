# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:09:18 2024
@author: haris
comment1
"""
import os
import cv2


fgModel = cv2.createBackgroundSubtractorMOG2()
leastNumOfFrames = 5
idx = []
C = []



#video_to_process = "space_traffic.mp4"
video_to_process = "outdoor.mp4"

#%% Create folder to store frame and binary segmented image
filename = os.path.splitext(video_to_process)[0]
print(filename)
folder_out1 = filename + '_frame'
folder_out2 = filename + '_binary'

if not os.path.exists(folder_out1):
    os.makedirs(folder_out1)

if not os.path.exists(folder_out2):
    os.makedirs(folder_out2)    

#%%

captured_video = cv2.VideoCapture(video_to_process)

i=0
while True:
    # read video frames
    retval, frame = captured_video.read()

    # check whether the frames have been grabbed
    if not retval:
        break

    # resize video frames
    frame = cv2.resize(frame, (640, 360))

    # pass the frame to the background subtractor
    fgmask = fgModel.apply(frame)
 
    # show the current frame, foreground mask, subtracted result
    cv2.imshow("Frames", frame)
    # show segmented foreground as binary image 
    cv2.imshow("Foreground Masks", fgmask)
    
    file_frame = os.path.join(folder_out1, 'Frame'+str(i) +'.jpg')
    cv2.imwrite(file_frame, frame)
    file_frame = os.path.join(folder_out2, 'Frame'+str(i) +'.jpg')
    cv2.imwrite(file_frame, fgmask)
    
    i = i+1
    keyboard = cv2.waitKey(10)
    if keyboard == 27:
        break

captured_video.release() 
cv2.destroyAllWindows() 