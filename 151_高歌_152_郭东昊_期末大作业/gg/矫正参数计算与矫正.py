import cv2
import numpy as np

# 读取左右摄像头的图像
imgL = cv2.imread('left.jpg',0)
imgR = cv2.imread('right.jpg',0)

# 使用OpenCV的stereoRectify函数进行立体矫正
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

# 显示矫正后的图像
cv2.imshow('Disparity', disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()


/*
numDisparities 这个参数决定了在计算视差图时 对于每个像素点 算法会搜索多大的视差范围 一般来说 如果你的摄像头的焦距较长 或者你的双目摄像头的基线距离较大 那么你可能需要设置一个较大的numDisparities值 相反 如果焦距较短或者基线距离较小 那么你可能需要设置一个较小的numDisparities值 你可以先设置一个初步的值 然后逐渐调整 直到得到满意的结果 
blockSize 这个参数决定了在计算视差图时 对于每个像素点 算法会在多大的邻域内进行匹配 一般来说 如果你的图像的纹理信息丰富 那么你可以设置一个较小的blockSize值 这样可以获取到更多的细节信息 相反 如果你的图像的纹理信息较少 那么你可能需要设置一个较大的blockSize值 这样可以增加匹配的稳定性 你也可以先设置一个初步的值 然后逐渐调整 直到得到满意的结果 
*/
