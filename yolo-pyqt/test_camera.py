import cv2

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取摄像头帧
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("无法读取帧")
        break

    # 显示帧
    cv2.imshow('frame', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()