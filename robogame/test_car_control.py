# test_car_control.py
import cv2
import time
from gdhsyyds import move, stop, fire, capture_and_process, aim, rush_and_catch

AIM_FAILED = 0
AIM_SUCCEED = 1

INVOLVE_TIME = 2  # 方块进入车内的时间，单位s
FIRE_TIME = 2  # 发射时间，单位s

def test_serial_movement():
    """
    测试小车的串口信号发送，前后左右移动和发射。
    """
    # 测试前进、中速
    move('W', 'S')
    time.sleep(1)
    stop()

    # 测试后退、低速
    move('S', 'C')
    time.sleep(1)
    stop()

    # 测试向左、高速
    move('A', 'C')
    time.sleep(1)
    stop()

    # 测试向右、低速
    move('D', 'C')
    time.sleep(1)
    stop()

    # 测试发射近距离
    fire(50)
    time.sleep(2)

    # 测试发射远距离
    fire(150)
    time.sleep(2)


def test_camera(cam_id):
    """
    测试摄像头的捕捉和处理功能，并显示处理结果。
    """
    for i in range(10):  # 读取10帧来进行测试
        vector = capture_and_process(cam_id)
        if vector is not None:
            print(f"摄像头 {cam_id} 捕捉到方块的相对坐标: {vector}")
        else:
            print(f"摄像头 {cam_id} 未检测到方块")

    # 关闭显示窗口
    cv2.destroyAllWindows()


def test_aim():
    """
    测试 aim 模块，检测小车是否能够成功瞄准方块。
    """
    result, cam_id = aim()
    if result == AIM_SUCCEED:
        print(f"瞄准成功，使用摄像头: {cam_id}")
    else:
        print("瞄准失败")


def test_rush_and_catch():
    """
    测试 rush_and_catch 模块，检测小车是否能够成功突进并捕获方块。
    """
    cam_id = 0  # 假设瞄准使用摄像头 0
    rush_and_catch(cam_id)
    print("突进并捕获完成")


if __name__ == "__main__":
    # 选择性运行测试
    test_serial_movement()   # 测试串口移动
    # test_camera(0)           # 测试摄像头 0
    # test_camera(1)           # 测试摄像头 1
    # test_aim()               # 测试瞄准功能
    # test_rush_and_catch()     # 测试突进和捕获功能
