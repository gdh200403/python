import cv2
import numpy as np
import time
import serial

# 设置串口参数
ser = serial.Serial('/dev/ttyUSB0', 115200)  # 选择串口（Windows: 'COM3' 等，Linux: '/dev/ttyUSB0' 等）

# 摄像头捕获和处理照片的函数
def capture_and_process(cam_id):
    """
    从指定摄像头ID捕获图像并处理，提取橙色方块中离摄像头最近的方块的方向向量 (x, y)
    
    参数:
    cam_id: 摄像头ID，0 或 1 表示不同的摄像头
    
    返回:
    (center_x, center_y), distance_vector: 最近方块的中心点坐标与相对摄像头的距离向量，如果未找到方块则返回 None
    """
    # 打开摄像头
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {cam_id}")
        return None, None
    
    # 读取图像
    ret, frame = cap.read()
    if not ret:
        print(f"无法捕获摄像头 {cam_id} 图像")
        return None, None
    
    # 转换为HSV并处理图像以找到橙色方块
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # 形态学操作去噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化最近方块的距离和位置
    min_distance = float('inf')
    nearest_block_center = None
    camera_focal_length = 800  # 假设的焦距，实际需要校准
    real_block_size = 10  # 方块的真实大小，单位为厘米
    
    # 遍历每个轮廓，找出矩形框并计算距离
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算矩形的中心点
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])
        
        # 计算矩形的宽度和高度（长边和短边）
        width = rect[1][0]
        height = rect[1][1]
        
        # 假设宽度是方块的实际边长，计算距离
        block_width = max(width, height)
        distance = (real_block_size * camera_focal_length) / block_width
        
        # 判断是否是最近的方块
        if distance < min_distance:
            min_distance = distance
            nearest_block_center = (center_x, center_y)
        
        # 在图像上绘制矩形框和中心点
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    if nearest_block_center is not None:
        # 计算距离向量（假设摄像头在图像中心）
        image_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        distance_vector = (nearest_block_center[0] - image_center[0], nearest_block_center[1] - image_center[1])
        
        # 显示处理后的图像
        cv2.imshow(f"Camera {cam_id} - Detected Blocks", frame)
        cv2.waitKey(1)  # 非阻塞显示
        
        return nearest_block_center, distance_vector
    else:
        return None, None

# 根据方向向量生成运动信号的函数
def generate_motion_signal(x, y, move_type):
    """
    根据摄像头方向向量 (x, y) 生成运动信号，依次输出左右和前后移动信号
    
    参数:
    x, y: 摄像头的方向向量
    move_type: '左右' 或 '前后' 移动类型
    
    返回:
    motion_signal_str: 生成的运动信号字符串
    """
    motion_signal = ['0'] * 6  # 初始化所有字节为 '0'
    motion_signal[0] = '1'  # 第0字节：1表示可以运动

    # 第1字节：左右或前后移动控制
    if move_type == '左右':
        if y > 0:
            motion_signal[1] = 'D'  # 向右
        elif y < 0:
            motion_signal[1] = 'A'  # 向左
    elif move_type == '前后':
        if x > 0:
            motion_signal[1] = 'W'  # 向前
        elif x < 0:
            motion_signal[1] = 'S'  # 向后

    # 第2字节：速度控制
    if move_type == '左右':
        motion_signal[2] = 'S'  # Shift 加速左右移动
    elif move_type == '前后':
        motion_signal[2] = 'C'  # CTRL 减速前后移动

    # 第3字节：1表示启动发射
    motion_signal[3] = '0'

    # 第4字节：1表示低速发射
    motion_signal[4] = '1'

    # 第5字节：0表示结束位
    motion_signal[5] = '0'

    signal_str = ''.join(motion_signal)
    ser.write(signal_str.encode())  # 通过串口发送信号
    print(f"发送的运动信号: {signal_str}")
    return signal_str

# 主流程控制函数
def main():
    """
    主函数控制流程，获取摄像头方向向量并生成运动信号，依次输出左右和前后移动
    """
    vec1, dist_vec1 = capture_and_process(0)  # 摄像头1
    vec2, dist_vec2 = capture_and_process(1)  # 摄像头2

    if vec1 is not None and vec2 is None:
        (x, y) = dist_vec1
    elif vec2 is not None and vec1 is None:
        (x, y) = - dist_vec2
    elif vec1 is not None and vec2 is not None:
        (x1, y1) = dist_vec1
        (x2, y2) = dist_vec2
        if x1 + 2*y1 < x2 + 2*y2:
            (x, y) = (x1, y1)
        else:
            (x, y) = (- x2, - y2)
    else:
        print("未能从摄像头获取方向向量")
        return

    # 计算左右和前后移动的时间
    left_right_speed = 30  # 左右移动速度，单位cm/s
    forward_backward_speed = 20  # 前后移动速度，单位cm/s
    wait_time = 1  # 等待时间，单位s
    
    left_right_time = abs(y) / left_right_speed  # 左右移动所需时间
    forward_backward_time = abs(x) / forward_backward_speed  # 前后移动所需时间

    # 输出左右移动信号
    motion_signal_lr = generate_motion_signal(x, y, '左右')
    print(f"输出左右移动信号: {motion_signal_lr}")
    
    # 等待左右移动完成
    time.sleep(left_right_time + wait_time)
    
    # 输出前后移动信号
    motion_signal_fb = generate_motion_signal(x, y, '前后')
    print(f"输出前后移动信号: {motion_signal_fb}")
    
    # 等待前后移动完成
    time.sleep(forward_backward_time + wait_time)

    signal_shoot = '0GG110'
    ser.write(signal_shoot.encode())  # 发射信号



# 执行主函数
if __name__ == "__main__":
    main()
