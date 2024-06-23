import cv2

# 读取双目摄像头的输入图片
img = cv2.imread('stereo_image.jpg')

# 获取图片的宽度和高度
height, width = img.shape[:2]

# 分割图片为两个图片
img_left = img[:, :width//2]
img_right = img[:, width//2:]

# 分别保存两个图片
cv2.imwrite('left_image.jpg', img_left)
cv2.imwrite('right_image.jpg', img_right)