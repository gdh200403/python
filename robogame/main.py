import cv2
import numpy as np
import time
import serial
import random
import signal
import sys
# import binascii

CAMERA_FOCAL_LENGTH = 800  # 像素焦距
REAL_BLOCK_SIZE = 10  # 方块的真实大小，单位为厘米

AIM_FAILED = 0 
AIM_SUCCEED = 1

INVOLVE_TIME = 2  # 方块进入车内的时间，单位s
FIRE_TIME = 3  # 发射时间，单位s

LOCAL_0 = 150
SPEED_LR_C = 30 # 左右慢速移动速度，单位为cm/s

# 设置串口参数
ser = serial.Serial('/dev/ttyUSB0', 115200)  # 选择串口（Windows: 'COM3' 等，Linux: '/dev/ttyUSB0' 等）
ser_1 = serial.Serial('/dev/ttyUSB1', 115200)  # 选择串口（Windows: 'COM3' 等，Linux: '/dev/ttyUSB0' 等）

flag_run=True

def signal_handler(signal, frame):
    print('Caught Ctrl+C / SIGINT signal')
    global flag_run
    flag_run=False


signal.signal(signal.SIGINT, signal_handler)

# 打开摄像头
# cap0 = cv2.VideoCapture(0)
    
# if not cap0.isOpened():
#     print(f"cannot open camera {0}")
# else:
#     cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# cap1 = cv2.VideoCapture(2)
    
# if not cap1.isOpened():
#     print(f"cannot open camera {1}")
# else:
#     cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 全局变量用于统计车左移和右移的时间
left_move_time = 0
right_move_time = 0

# 摄像头捕获和处理照片的函数
def capture_and_process(cam_id):
    """
    从指定摄像头ID捕获图像并处理，提取橙色方块中离摄像头最近的方块的方向向量 (x, y)
    
    参数:
    cam_id: 摄像头ID，0 或 1 表示不同的摄像头
    
    返回:
    pseudo_distance_vector: 最近方块的中心点坐标与相对摄像头的伪距离向量，如果未找到方块则返回 None
    """
    
    # 读取图像
    if(cam_id == 0):
        # cap = cap0
        cap = cv2.VideoCapture(0)
    elif(cam_id == 1):
        # cap = cap1
        cap = cv2.VideoCapture(2)
    else:
        print("camera id error")
        return
    
    ret, frame = cap.read()
    if not ret:
        print(f"cannot capture image from camera {cam_id} ")
        return None
    
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
    camera_focal_length = CAMERA_FOCAL_LENGTH  # 假设的焦距，实际需要校准
    real_block_size = REAL_BLOCK_SIZE  # 方块的真实大小，单位为厘米
    
    # 遍历每个轮廓，找出矩形框并计算距离
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
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
        pseudo_distance_vector = (nearest_block_center[0] - image_center[0], nearest_block_center[1])
        
        # 显示处理后的图像
        # cv2.imshow(f"Camera {cam_id} - Detected Blocks", frame)
        # cv2.waitKey(1)  # 非阻塞显示
        
        cap.release()

        return pseudo_distance_vector
    else:
        return None

def move(direction, speed):
    global ser
    """
    控制小车移动的函数
    
    参数:
    direction: 移动方向，'W' 表示前进，'S' 表示后退，'A' 表示向左，'D' 表示向右, '0' 表示停止
    speed: 移动速度，'S' 表示高速，'G' 表示中速，'C' 表示低速
    """

    motion_signal = ['0'] * 6  # 初始化所有字节为 '0'
    motion_signal[0] = '1'  # 第0字节：1表示可以运动
    
    # 第1字节：移动控制
    motion_signal[1] = direction
    
    # 第2字节：速度控制
    motion_signal[2] = speed
    
    # 第3字节：1表示启动发射
    motion_signal[3] = '0'
    
    # 第4字节：0表示摩擦轮不转
    motion_signal[4] = '0'
    
    # 第5字节：0表示结束位
    motion_signal[5] = '0'
    
    signal_str = ''.join(motion_signal)

    ser.write(signal_str.encode())  # 通过串口发送信号
    print(f"direction: {direction}, speed: {speed}")


def stop():
    global ser
    """
    控制小车停止的函数
    """
    signal_str = '0GG000'
    ser.write(signal_str.encode())  # 通过串口发送信号
    time.sleep(1)
    print(f"stop")

def fire(shoot_distance):
    global ser
    """
    控制小车发射的函数
    
    参数:
    shoot_distance: 发射距离，单位为厘米
    """
    is_need_shoot_faster = shoot_distance > 150  # 距离大于150cm时使用高速发射

    SIG_SHOOT_FAST = '0GG120'
    SIG_SHOOT_SLOW = '0GG110'

    if is_need_shoot_faster:
        ser.write(SIG_SHOOT_FAST.encode())
        print(f"fire fast")
    else:
        ser.write(SIG_SHOOT_SLOW.encode())
        print(f"fire slow")

def shoot_distance():
    global left_move_time, right_move_time
    
    local = 0
    local =(- right_move_time + left_move_time) * SPEED_LR_C + LOCAL_0

    return local

def choose_with2eyes():
    '''
    功能：前后两个摄像头搜索目标
    返回：选择的目标的伪方向向量，(x, y)，以及发现它的摄像头的ID
    注意此时(x,y)已经转换为不论前后，已经是前进方向为y正方向，向右为x正方向的坐标系
    '''

    vec1 = capture_and_process(0)  # 摄像头1
    vec2 = capture_and_process(1)  # 摄像头2

    if vec1 is not None and vec2 is None:
        (x, y) = vec1
        return (x, y), 0
    elif vec2 is not None and vec1 is None:
        (x, y) = (-vec2[0],-vec2[1])
        return (x, y), 1
    elif vec1 is not None and vec2 is not None:
        (x1, y1) = vec1
        (x2, y2) = vec2
        if 3*abs(x1) + y1 < 3*abs(x2) + y2:
            (x, y) = (x1, y1)
            return (x, y), 0
        else:
            (x, y) = (- x2, - y2)
            return (x, y), 1
    else:
        print("cannot get direction vector from the camera")
        return None, None

def chase_with1eye(eye):
    '''
    功能：单独一个摄像头搜索目标，节省资源
    返回：选择的目标的伪方向向量，(x, y)
    '''
    for _ in range(5):
        eye.read()

    ret, frame = eye.read()
    if not ret:
        print(f"cannot capture image from camera {cam_id} ")
        return None
    
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
    camera_focal_length = CAMERA_FOCAL_LENGTH  # 假设的焦距，实际需要校准
    real_block_size = REAL_BLOCK_SIZE  # 方块的真实大小，单位为厘米
    
    # 遍历每个轮廓，找出矩形框并计算距离
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
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
        pseudo_distance_vector = (nearest_block_center[0] - image_center[0], nearest_block_center[1])
        
        # 显示处理后的图像
        # cv2.imshow(f"Camera {cam_id} - Detected Blocks", frame)
        # cv2.waitKey(1)  # 非阻塞显示

        return pseudo_distance_vector
    else:
        return None

def wander():
    '''
    功能：在choose_with2eyes函数找不到方块时被调用，随机前后缓慢移动适当距离
    '''
    global left_move_time, right_move_time

    random_number = random.random()
    if shoot_distance()<=30:
        move('A', 'G')
        start_time = time.time()
        time.sleep(2)
        stop()
        left_move_time += time.time() - start_time
    elif shoot_distance()>=270:
        move('D', 'G')
        start_time = time.time()
        time.sleep(2)
        stop()
        right_move_time += time.time() - start_time
    elif random_number < 0.5:
        move('A', 'G')
        start_time = time.time()
        time.sleep(2)
        stop()
        left_move_time += time.time() - start_time
    else:
        move('D', 'G')
        start_time = time.time()
        time.sleep(2)
        stop()
        right_move_time += time.time() - start_time
    return

def aim():
    '''
    功能：找寻距离最近的方块，进行左右移动
    移动过程中持续拍照检测车与该方块的左右距离
    检测到该距离为一个较小值的时候，停止移动
    返回值：瞄准是否成功(0失败，1成功), 瞄准方块的摄像头编号cam_id
    '''
    global left_move_time, right_move_time

    LR_DISTANCE_THRESHOLD = 20  # 设定的距离阈值，单位px
    print("searching for nearest block")
    vec, cam_id = choose_with2eyes()
    while cam_id is None:
        wander()
        vec, cam_id = choose_with2eyes()
        print("searching for nearest block")
    # if cam_id == 0: # 选择的是摄像头1，即前进方向的摄像头

    lr_distance = vec[0]
    is_aimed = False

    start_time = time.time()
    is_left = False

    if(cam_id == 0):
        working_eye = cv2.VideoCapture(0)
    else:
        working_eye = cv2.VideoCapture(2)

    while not is_aimed and flag_run:
        print("aiming for the block")
        if abs(lr_distance) < LR_DISTANCE_THRESHOLD:
            is_aimed = True
            # stop()
        else:
            if lr_distance < 0:
                move('A', 'S') 
                start_time = time.time()
                time.sleep(0.5)
                stop()
                is_left = True
                # left_move_time += time.time() - start_time
            else:
                move('D', 'S')
                start_time = time.time()
                time.sleep(0.5)
                stop()
                is_left = False
                # right_move_time += time.time() - start_time
            
            vec = chase_with1eye(working_eye)
            if vec is None: # 丢失目标的简单错误处理
                stop()
                return AIM_FAILED, cam_id
            lr_distance = vec[0]
            
        if(is_left):
            left_move_time += time.time() - start_time
        else:
            right_move_time += time.time() - start_time

    working_eye.release()
    
    return AIM_SUCCEED, cam_id

def rush_and_catch(cam_id):
    '''
    功能：在aim成功后，突进至方块面前，在减速点减速，在减速运动一定时间后停止，此时方块已经被收入囊中
    '''           
    DECELERATION_DISTANCE = 100  # 减速距离，单位px
    DECELERATION_TIME = 1.5  # 减速运动时间，单位s
    
    if(cam_id == 0):
        working_eye = cv2.VideoCapture(0)
    else:
        working_eye = cv2.VideoCapture(2)

    vec = chase_with1eye(working_eye)

    if vec is None:
        stop()
        return

    y = vec[1]
    fb_distance = abs(y)
    while fb_distance > DECELERATION_DISTANCE:
        move('W' if cam_id == 0 else 'S', 'S')
        vec = chase_with1eye(cam_id)
        if vec is None:
            stop()
            return
        y = vec[1]
        fb_distance = abs(y)

    working_eye.release()
    move('W' if cam_id == 0 else 'S', 'C')
    time.sleep(DECELERATION_TIME)
    stop()
    return
    
def get_distance_inside():
    '''
    功能：获取车内方块的位置信息
    '''
    global ser_1
    try:
        ser_1.reset_input_buffer()
        time.sleep(0.1)  # 等待一段时间以确保接收到最新的数据
        num = ser_1.inWaiting()
        data = ser_1.read(num)
        result = 0
        i = 3
        while i < 7:
            result = result * 16 + data[i]
            i += 1
        
        print(result)
        return result
    except: #--则将其作为字符串读取
        pass

    return None


def is_block_involved():
    '''
    功能：判断车内是否有方块
    '''
    INVOLVED_DISTANCE_THRESHOLD = 35000  # 设定的阈值，单位cm
    distance_inside = get_distance_inside()
    if distance_inside is None:
        return False  # 如果 distance_inside 是 None，返回 False
    
    if distance_inside < INVOLVED_DISTANCE_THRESHOLD:
        return True
    else:
        return False

# 将十六进制字符转换为十进制数
def change(letter):

    if letter == 'a':
        letter = 10
    elif letter == 'b':
        letter = 11
    elif letter == 'c':
        letter = 12
    elif letter == 'd':
        letter = 13
    elif letter == 'e':
        letter = 14
    elif letter == 'f':
        letter = 15
        
    return int(letter)


# 主流程控制函数
def main():
    stop()
    # move('D','G')
    # time.sleep(2)
    # stop()
    while flag_run:
        # 寻找最近方块，并通过左右移动瞄准方块，如果瞄准失败则重新寻找并再次瞄准
        aim_result, cam_id = aim()
        if aim_result == AIM_FAILED:
            print("aim failed, continue to search")
            continue

        # 突进并捕获方块
        print("aim succeed, rush and catch")
        rush_and_catch(cam_id)
        
        # 等待方块进入车内
        print("waiting for block to be involved")
        time.sleep(INVOLVE_TIME) 

        # 判断车内是否有方块，有则发射，无则可能出了点问题，继续寻找
        # if is_block_involved():
        #     print("block involved, fire")
        #     fire(shoot_distance())
        # else:
        #     continue

                # 判断车内是否有方块，有则发射，无则可能出了点问题，继续寻找
        if is_block_involved():
            if shoot_distance()>150:
                move('D','G')
                time.sleep(2)
                stop()
            elif shoot_distance()<50:
                
                move('A','G')
                time.sleep(2)
                stop()
            fire(shoot_distance())
        else:
            continue

        time.sleep(FIRE_TIME)
    ser.close()
    ser_1.close()
    sys.exit(0)


# 执行主函数
if __name__ == "__main__":
    main()
