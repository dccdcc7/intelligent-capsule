import cv2
import threading
import time
class VideoPlayer:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.is_paused = False
        self.is_running = True

    def play(self):
        threading.Thread(target=self.thread_function).start()

    def thread_function(self):
        while self.is_running:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Reached the end of the video or failed to read the frame.")
                    self.is_running = False
                    break

                cv2.imshow('Video Player', frame)
                if cv2.waitKey(30) & 0xFF == ord('p'):
                    self.is_paused = not self.is_paused
                elif cv2.waitKey(30) & 0xFF == ord('q'):
                    self.is_running = False
                    break
            else:
                cv2.waitKey(30)

        # Release everything if job is finished
        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.is_running = False

# 使用VideoPlayer类
video_source = 0  # 使用0通常表示默认的摄像头，如果是视频文件则替换为文件路径
player = VideoPlayer(video_source)
player.play()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    player.stop()