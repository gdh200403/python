# demo.py
# 效果：不太看得出来的散点图生成
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from download import download_dataset


# 要下载的数据集名称
dataset_name = "skates1"

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

    # 使用cv2.findFundamentalMat计算得到的基础矩阵M
    # 例如：M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)

    # 从基础矩阵计算本征矩阵
    E = K1.T @ M @ K0  # 使用@进行矩阵乘法

    # 从本征矩阵恢复旋转和平移信息
    retval, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K0)

    # 准备立体校正参数
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K0, np.zeros(5), K1, np.zeros(5), img1.shape[::-1], R, t)

    # 三角测量计算匹配点的三维坐标
    points4D = cv2.triangulatePoints(P1, P2, src_pts, dst_pts)

    # 将齐次坐标转换为3D坐标
    points3D = points4D[:3] / points4D[3]

    # 可视化三维点云

    # 创建一个新的图形和3D轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将3D点的坐标分开以便绘制
    X = points3D[0]
    Y = points3D[1]
    Z = points3D[2]

    # 绘制散点图
    ax.scatter(X, Y, Z)

    # 设置轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 显示图表
    plt.show()

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None