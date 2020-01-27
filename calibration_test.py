import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

cm = np.load('calibration_matrix.npz')

K = cm['K']
D = cm['D']
dim = cm['dim']
# print(K, D, dim)
# You should replace these 3 lines with the output in calibration step
# DIM = (320, 427)
# K=np.array(YYY)
# D=np.array(ZZZ)
# calibration_matrix = {'K':K, 'D':D, 'DIM':(427, 320)}
# np.save('calibration_matrix.npy', calibration_matrix)
path = '/home/siyuan/Documents/fisheye_calibration/data/'

for i in range(1,1000):
    img = cv2.imread(path+str(i)+'.jpg')
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (427, 320), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

