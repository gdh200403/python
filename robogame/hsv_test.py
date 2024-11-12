import cv2
import numpy as np

def test_orange_block_detection(cam_id=0):
    # 打开摄像头q
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {cam_id}")
        return
    
    while True:
        # 读取图像
        ret, frame = cap.read()
        if not ret:
            print(f"无法捕获摄像头 {cam_id} 图像")
            break
        
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
        
        # 在图像上绘制轮廓
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # 使用 int32 代替 int0
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            
            # 计算矩形的中心点
            center_x = int(rect[0][0])
            center_y = int(rect[0][1])
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 显示处理后的图像
        cv2.imshow(f"Camera {cam_id} - Detected Orange Blocks", frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 测试橙色方块检测（假设摄像头ID为0）
test_orange_block_detection(1)