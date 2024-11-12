import cv2
import time

def check_external_camera(cam_id=1):
    start_time = time.time()
    
    # 尝试打开指定的摄像头
    cap = cv2.VideoCapture(cam_id)
    open_time = time.time()
    
    # 检查摄像头是否成功打开
    if cap.isOpened():
        print(f"摄像头 {cam_id} 连接成功")
        
        # 获取当前分辨率
        current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"当前分辨率: {current_width} x {current_height}")
        get_resolution_time = time.time()
        
        # 读取一帧图像
        ret, frame = cap.read()
        read_frame_time = time.time()
        
        if ret:
            # 显示图像
            cv2.imshow(f"Camera {cam_id} - Captured Image", frame)
            cv2.waitKey(0)  # 等待按键按下
            cv2.destroyAllWindows()  # 关闭窗口
        else:
            print(f"无法从摄像头 {cam_id} 捕获图像")
        
        cap.release()  # 释放摄像头资源
        release_time = time.time()
        
        # 打印每一步的耗时
        print(f"打开摄像头耗时: {open_time - start_time:.4f} 秒")
        print(f"获取分辨率耗时: {get_resolution_time - open_time:.4f} 秒")
        print(f"读取图像耗时: {read_frame_time - get_resolution_time:.4f} 秒")
        print(f"释放摄像头耗时: {release_time - read_frame_time:.4f} 秒")
        
        return True
    else:
        print(f"无法连接摄像头 {cam_id}")
        return False

# 检测外接摄像头（假设外接摄像头的ID为1）
check_external_camera(1)