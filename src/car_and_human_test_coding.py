# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:41:23 2024

@author: sharm
"""

import os
import cv2
import numpy as np


#from skimage.segmentation import clear_border
#from skimage.measure import label, regionprops


#%%
# Create a background subtractor
fgModel = cv2.createBackgroundSubtractorMOG2()
leastNumOfFrames = 3
idx = []
C = []

# Video file to process
#video_to_process = "car test.mp4" & 'car and human test video'

video_to_process = "Indoor.mp4"


#%%
def function1():
    print(0)


#%%
# Create folder to store frames and binary segmented images
filename = os.path.splitext(video_to_process)[0]
print(filename)
folder_out1 = filename + '_frame'
folder_out2 = filename + '_binary'

if not os.path.exists(folder_out1):
    os.makedirs(folder_out1)

if not os.path.exists(folder_out2):
    os.makedirs(folder_out2)

# Structuring element for morphological operations
kernel = np.ones((3, 3), np.uint8)  # 5x5 kernel for opening (adjust size if needed)

# Capture the video
captured_video = cv2.VideoCapture(video_to_process)

i = 0
while True:
    # Read video frames
    retval, frame = captured_video.read()

    # Check if the frames have been grabbed
    if not retval:
        break

    # Resize video frames
    frame = cv2.resize(frame, (640, 480))

    # Pass the frame to the background subtractor
    fgmask = fgModel.apply(frame)

    # Apply morphological opening to remove noise (erosion followed by dilation)
    fgmask_opened = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours (connected components) in the binary mask
    contours, _ = cv2.findContours(fgmask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through each contour to detect and draw bounding boxes
    for contour in contours:
        # Filter out small regions by area
        area = cv2.contourArea(contour)
        #read the dimension of the area
        x, y, w, h = cv2.boundingRect(contour)  
        #filter out those small area
        if (area>=550):
            #width > height =car
            if(w>h):
                label='human'
                color=(255, 255, 255)# White text
            elif(w<=h):
                #width < height = human/motorbike
                label='Human'
                color=(255, 255, 0)# Blue text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box
                
            # Define label text and font
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_size, _ = cv2.getTextSize(label, font, 0.5, 2)

            # Draw label above the bounding box
            label_x = x
            label_y = y - 10 if y - 10 > 10 else y + 10  # Ensure label is within frame
            cv2.putText(frame, label, (label_x, label_y), font, 0.5, color, 2) 
            
            
    # Show the current frame with bounding boxes and labels, and foreground mask
    cv2.imshow("Frames", frame)
    cv2.imshow("Foreground Masks", fgmask_opened)

    # Save the frame and the binary mask (after morphological opening)
    file_frame = os.path.join(folder_out1, 'Frame' + str(i) + '.jpg')
    cv2.imwrite(file_frame, frame)
    file_mask = os.path.join(folder_out2, 'Frame' + str(i) + '.jpg')
    cv2.imwrite(file_mask, fgmask_opened)
    #my_function('Frame{i}')
    i += 1
    keyboard = cv2.waitKey(15)
    if keyboard == 17:  
        break

captured_video.release()
cv2.destroyAllWindows()
