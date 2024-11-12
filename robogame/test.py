import cv2
import numpy as np

# 购买的摄像头参数
# camera_height = 26.1 / 100  # 摄像头离地面的高度，单位米
# tilt_angle = 58.6  # 摄像头的俯角，单位度
# sensor_size = (5.76, 4.29)  # 感光尺寸，单位毫米
# focal_length = 3.6  # 摄像头物理焦距，单位毫米
# resolution = (2594, 1944)  # 摄像头的分辨率

# 购买的摄像头参数
camera_height = 26.1 / 100  # 摄像头离地面的高度，单位米
tilt_angle = 45  # 摄像头的俯角，单位度
sensor_size = (13.36,10.02)  # 感光尺寸，单位毫米
focal_length = 6.04  # 摄像头物理焦距，单位毫米
resolution = (4000,3000)  # 摄像头的分辨率

# 颜色范围（橙色检测范围）
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])

# 计算水平视角和垂直视角
def get_fov(sensor_size, focal_length):
    return 2 * np.arctan(sensor_size / (2 * focal_length)) * 180 / np.pi

horizontal_fov = get_fov(sensor_size[0] / 1000, focal_length)  # 水平视角
vertical_fov = get_fov(sensor_size[1] / 1000, focal_length)  # 垂直视角

# 物理坐标计算函数
def calculate_physical_position(cx, cy):
    image_width, image_height = resolution
    
    # 中心坐标偏移
    delta_x = cx - image_width / 2
    delta_y = cy - image_height / 2
    
    # 像素对应的视角
    angle_x = delta_x * (horizontal_fov / image_width)
    angle_y = delta_y * (vertical_fov / image_height)

    # 计算垂直方向的投影距离 dy
    dy = camera_height / np.tan(np.radians(tilt_angle - angle_y))

    # 调整垂直投影距离
    d_adjusted = dy / np.cos(np.radians(tilt_angle))

    # 计算水平方向的位移 dx
    dx = d_adjusted * np.tan(np.radians(angle_x))

    # 计算总的水平距离 d
    d = np.sqrt(dy**2 + dx**2)

    return d, dx, dy

# 橙色方块检测并计算质心函数
def detect_cube_center(image_path):
    # 加载图像并转换为HSV
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 检测橙色区域
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("没有检测到橙色物体")
        return None
    
    # 找到最大的轮廓，假设它是方块
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        print("没有有效的轮廓")
        return None
    
    # 计算质心
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # 可视化结果
    result_image = image.copy()
    cv2.circle(result_image, (cx, cy), 5, (0, 255, 0), -1)
    cv2.putText(result_image, f'cx: {cx}, cy: {cy}', (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 显示图像
    cv2.imshow("Cube Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 转换成实际物理坐标
    return calculate_physical_position(cx, cy)

# 示例调用
image_path = "image.jpg"  # 替换为实际图片路径
position = detect_cube_center(image_path)

if position:
    d, dx, dy = position
    print(f"距离: {d:.2f}米, x方向位移: {dx:.2f}米, y方向位移: {dy:.2f}米")
