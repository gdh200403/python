import numpy as np
import cv2
import glob

# 标定图像的路径
images = glob.glob('opencv官方标定板/*.jpg')

# 设置标定板角点（inner corner）的数量
corner_x = 9  # OpenCV官方标定板的内角点数量
corner_y = 6

# 设置角点在世界坐标系中的坐标
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# 存储实际的三维点和图像中的二维点
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y),None)

    # 如果找到，添加对象点，图像点（并优化）
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 画出角点并显示
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)