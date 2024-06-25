# demo1.py
# 效果：输出深度图
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from download import download_dataset


# 要下载的数据集名称
dataset_name = "pendulum1"

def read_calibration(file_path):
    """
    从文件中读取相机标定参数
    :param file_path: 标定文件路径
    :return: 相机内参矩阵和其他参数的字典
    """
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.split('=')
            key = key.strip()
            if key in ['cam0', 'cam1']:
                # 更准确地处理矩阵字符串
                # 移除左右方括号并分割
                mat_values = value.strip()[1:-1].replace(';', '').split()
                # 转换为浮点数并重塑为3x3矩阵
                mat = np.array([float(num) for num in mat_values]).reshape(3, 3)
                params[key] = mat
            else:
                # 对于非矩阵值，直接转换并存储
                params[key] = float(value.strip())
    return params

# 调用函数开始下载数据集
data_dir = download_dataset(dataset_name)

# 读取标定参数
calib_file_path = os.path.join(data_dir, 'calib.txt')  # 假设标定文件在数据集目录下
calibration_data = read_calibration(calib_file_path)
print(calibration_data)

# 读取图像
img1_path = os.path.join(data_dir, 'im0.png')
img2_path = os.path.join(data_dir, 'im1.png')

img1 = cv2.imread(img1_path, 0)
img2 = cv2.imread(img2_path, 0)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 找到关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用FLANN匹配器
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 仅存储良好的匹配项
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# 计算基础矩阵
MIN_MATCH_COUNT = 10  # Define the value of MIN_MATCH_COUNT

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)

    # 只选择内点
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # 调整图像尺寸以适应屏幕
    screen_res = 1280, 720  # 假设屏幕分辨率为 1280x720
    scale_width = screen_res[0] / img3.shape[1]
    scale_height = screen_res[1] / img3.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img3.shape[1] * scale)
    window_height = int(img3.shape[0] * scale)

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Matches', window_width, window_height)

    cv2.imshow("Matches", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 从calibration_data提取相机内参矩阵
    K0 = calibration_data['cam0']  # 第一个相机的内参矩阵
    K1 = calibration_data['cam1']  # 第二个相机的内参矩阵

    # 1. 计算视差图
    # 创建立体匹配对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=int(calibration_data['vmin']),
        numDisparities=int(calibration_data['ndisp']),
        blockSize=5,  # 通常选择一个奇数作为块大小
        P1=8*3*5**2,  # 控制视差平滑度的参数
        P2=32*3*5**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 计算视差图
    disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0

    # 2. 计算深度图
    # 相机焦距 * 基线距离 / 视差
    focal_length = calibration_data['cam0'][0, 0]  # 假设焦距存储在内参矩阵的[0,0]位置
    baseline = calibration_data['baseline']

    # 为避免除以0导致错误，添加一个小的常数epsilon
    epsilon = 1e-5  
    depth = (focal_length * baseline) / (disparity + epsilon)

    # 3. 三维点的恢复
    # 创建重新投影矩阵
    h, w = img1.shape[:2]
    Q = np.float32([[1, 0, 0, -w/2.0],
                    [0, -1, 0, h/2.0],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1/baseline, 0]])

    # 重新投影到三维空间
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # 可视化深度图
    plt.imshow(depth, cmap='hot')
    plt.colorbar()
    plt.show()

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None