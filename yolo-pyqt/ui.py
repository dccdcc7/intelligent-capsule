from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time,yaml
from threading import Thread
from PyQt5.QtWidgets import QFileDialog,QApplication,QMessageBox
import sys
from yolo1 import test_yolov5_track2,yolov5
sys.path.append('F:/pycharmproject/segment-anything/yolo-pyqt')
from onnx11 import model_init,onnx_forward1
from yolo_utils.utils import non_max_suppression, letterbox, scale_coords, plot_one_box, detect_img
from ultralytics import YOLO, RTDETR

class MWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        # 设置界面
        self.setupUI()

        self.camBtn.clicked.connect(self.startCamera)
        self.videoBtn.clicked.connect(self.startVideo)
        self.stopBtn.clicked.connect(self.stop)

        # 定义定时器，用于控制显示视频的帧率
        self.timer_camera = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()
        # 定时到了，回调 self.show_camera
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_video.timeout.connect(self.show_video)
        # 加载 YOLO nano 模型，第一次比较耗时，要20秒左右
        #self.model = Yolov5ONNX('runs/train/exp27/weights/best.onnx')
        #self.model = Yolov5ONNX('runs/exp6/weights/best.onnx')
        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []

        with open('F:/pycharmproject/segment-anything/yolo-pyqt/yolov5s.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        self.yolo = yolov5(**cfg)
        self.yolo.track_init('ByteTrack')
        self.Model = model_init()
        self.frame_num = 0
        self.unwear_num = 0
        # 启动处理视频帧独立线程
        #Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()

    def setupUI(self):
        self.resize(700, 720)
        self.setWindowTitle('乘客状态检测系统')
        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)
        # 界面的上半部分 : 图形展示部分
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        self.label_ori_video.setMinimumSize(650,650)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        #self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')
        topLayout.addWidget(self.label_ori_video)
        #topLayout.addWidget(self.label_treated)
        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout =  QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)
        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn   = QtWidgets.QPushButton('📹摄像头')
        self.stopBtn  = QtWidgets.QPushButton('🛑停止')
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout)

    def get_file_path(self):
        # 创建QApplication实例
        app = QApplication(sys.argv)
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "所有文件 (*);;文本文件 (*.txt)")
        # 检查用户是否选择了文件
        if file_path:
            # 打印文件路径
            #print("选择的文件路径:", file_path)
            self.path = file_path
            return file_path
        else:
            QMessageBox.information(None, "提示", "未选择文件")
        # 退出应用程序
        sys.exit(app.exec_())

    def startCamera(self):
        # 参考 https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
        # 在 windows上指定使用 cv2.CAP_DSHOW 会让打开摄像头快很多，
        # 在 Linux/Mac上 指定 V4L, FFMPEG 或者 GSTREAMER
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("1号摄像头不能打开")
            return()

        if self.timer_camera.isActive() == False:  # 若定时器未启动
            self.timer_camera.start(50)
        self.frame_num += 1

    def startVideo(self):
        path = self.get_file_path()
        self.cap = cv2.VideoCapture(path)
        if self.timer_video.isActive() == False:  # 若定时器未启动
            self.timer_video.start(50)

    def show_camera(self):
        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return
        # 把读到的16:10帧的大小重新设置
        # if not self.frameToAnalyze:
        #     self.frameToAnalyze.append(frame)
        frame = cv2.resize(frame, (640, 640))
        #self.unwear_num = 0
        image, result = onnx_forward1(self.Model, frame.copy())
        # print(result)
        image, out_num, total_num = self.yolo.track_processing(self.frame_num,frame.copy(), result)
        if (out_num != 0):
            self.unwear_num += 1
        startppoint = int(image.shape[1] / 2)
        # print(startppoint)
        content2 = str(total_num) + " people on board"
        cv2.putText(image, content2, (startppoint - 100, image.shape[0] - 10), 0, 0.5, [0, 255, 0], thickness=2,
                    lineType=cv2.LINE_AA)
        if (self.unwear_num >= 100):
            print("Warning!!!!!!")
            content1 = "Warning!"
            cv2.putText(image, content1, (10, 10), 0, 0.5, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
            if (out_num == 0):
                self.unwear_num -= 30
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 变成QImage形式
        # 往显示视频的Label里 显示QImage
        #print("unwaer_num:",self.unwear_num)
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def show_video(self):
        list2=[]
        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return
        # 把读到的16:10帧的大小重新设置
        frame = cv2.resize(frame, (640, 640))
        self.unwear_num = 0
        image, result = onnx_forward1(self.Model, frame.copy())
        # print(result)
        image, out_num, total_num = self.yolo.track_processing(self.frame_num, frame.copy(), result)
        if (out_num != 0):
            self.unwear_num += 1
        startppoint = int(image.shape[1] / 2)
        # print(startppoint)
        content2 = str(total_num) + " people on board"
        #cv2.putText(image, content2, (startppoint - 300, image.shape[0] - 10), 0, 0.5, [0, 255, 0], thickness=2,
                    #lineType=cv2.LINE_AA)
        if (self.unwear_num >= 100):
            print("Warning!!!!!!")
            content1 = "Warning!"
            cv2.putText(image, content1, (10, 10), 0, 0.5, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
            if (out_num == 0):
                self.unwear_num -= 30
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                              QtGui.QImage.Format_RGB888)  # 变成QImage形式
        # 往显示视频的Label里 显示QImage
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def stop(self):
        self.timer_camera.stop()  # 关闭定时器
        self.cap.release()  # 释放视频流
        self.label_ori_video.clear()  # 清空视频显示区域
        self.label_treated.clear()  # 清空视频显示区域


if __name__ =='__main__':
    app = QtWidgets.QApplication()
    window = MWindow()
    window.textLog.setPlainText("欢迎使用人数检测系统，视频左上角显示的是实时人数")
    window.show()
    app.exec()



