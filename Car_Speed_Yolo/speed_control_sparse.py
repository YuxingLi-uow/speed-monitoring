import numpy as np
import cv2
from yolo_v3 import *

cap = cv2.VideoCapture('video_01.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p0 = np.array([])
old_gray = None
mask = None

filter = cv2.imread('mask.jpg')
filter = cv2.resize(filter, None, fx=0.4, fy=0.4)


# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


while(1):
    ret,frame = cap.read()
    frame_resized = cv2.resize(frame, None, fx=0.4, fy=0.4)
    frame_resized = cv2.bitwise_and(frame_resized, filter)  # apply mask
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    if old_gray is None:
        old_gray = frame_gray  # first frame, no detection
        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame_resized)
        for x1, y1, x2, y2, cls in detect_cars(frame_resized):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            p0 = np.append(p0, [center_x, center_y])  # as feature points
            p0 = np.reshape(p0, (-1, 1, 2))
        continue

    for x1, y1, x2, y2, cls in detect_cars(frame_resized):
        p0 = np.reshape(p0, (-1,))  # flat feature points
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        p0 = np.append(p0, [center_x, center_y])  # as feature points
        p0 = np.reshape(p0, (-1, 1, 2))
        p0 = np.unique(p0, axis=0).astype(np.float32)
        print(p0.shape)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i, (new,old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), 2)
            frame_resized = cv2.circle(frame_resized, (a, b), 5, -1)
        img = cv2.add(frame_resized,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()