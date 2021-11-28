import numpy as np
import cv2


def mask(image, pointsArray):

    mask = np.zeros(image.shape, np.uint8)
    pts_array = np.array(pointsArray, np.int32)
    pts_array.reshape((-1, 1, 2))
    mask = cv2.polylines(mask, [pts_array], True, (255, 255, 255))
    mask_2 = cv2.fillPoly(mask, [pts_array], (255, 255, 255))

    cv2.imshow('mask_2', mask_2)
    ROI = cv2.bitwise_and(mask_2, image)
    cv2.imshow('ROI', ROI)
    # cv2.imwrite('mask.jpg', mask_2)


img = cv2.imread('frame1.jpg')
mask(img, [[188, 160], [434, 160], [1088, 720], [0, 720]])

cv2.waitKey(0)
cv2.destroyAllWindows()

