import numpy as np
import cv2


img = cv2.imread("motion_bike.jpg", 0)

crop_img = img[0:335, 99:439]

bar = np.zeros((5,340),dtype = int)

print crop_img.shape

cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.destroyWindow('cropped') 
