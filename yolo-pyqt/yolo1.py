import os, random, yaml, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random.seed(0)
import cv2
import torch
import numpy as np
import sys
sys.path.append('F:/pycharmproject/segment-anything/yolo-pyqt')
from onnx11 import model_init,onnx_forward1
from yolo_utils.utils import non_max_suppression, letterbox, scale_coords, plot_one_box, detect_img
from ultralytics import YOLO, RTDETR
from PySide6 import QtWidgets, QtCore, QtGui

def ByteTrack_opt():
    parser = argparse.ArgumentParser("ByteTrack Param.")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=50, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fps", default=25, type=int, help="frame rate (fps)")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser.parse_args()

class base_model:
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs):
        device = self.select_device(device)
        print(device)
        if model_path.endswith('pt'):
            model = torch.jit.load(model_path).to(device)
        elif model_path.endswith('onnx'):
            try:
                import onnxruntime as ort
            except:
                raise 'please install onnxruntime.'
            providers = ['CUDAExecutionProvider'] if device.type != 'cpu' else ['CPUExecutionProvider']
            model = ort.InferenceSession(model_path, providers=providers)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        self.__dict__.update(locals())
    
    def __call__(self, data):
        if type(data) is str:
            image = cv2.imdecode(np.fromfile(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = data
        im = self.processing(image)
        
        if self.model_path.endswith('pt'):
            result = self.model(im)[0]
        elif self.model_path.endswith('onnx'):
            result = self.model.run([i.name for i in self.model.get_outputs()], {'images':im})[0]
        return self.post_processing(result, im, image)
        
    def processing(self, img):
        image = letterbox(img, new_shape=tuple(self.imgsz), auto=False)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        image = np.array(image, dtype=np.float32)
        image /= 255
        
        if self.model_path.endswith('pt'):
            im = torch.from_numpy(image).float().to(self.device)
        elif self.model_path.endswith('onnx'):
            im = image
        return im
    
    def post_processing(self, result, im=None, img=None):
        pass
    
    def select_device(self, device):
        if device == -1:
            return torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
            device = torch.device('cuda:0')
        return device
    
    def track_init(self, track_type):
        from track_utils.byte_tracker import BYTETracker, BaseTrack
        if track_type == 'ByteTrack':
            self.track_opt = ByteTrack_opt()
            self.tracker = BYTETracker(self.track_opt, frame_rate=self.track_opt.fps)
            BaseTrack._count = 0
    
    def track_processing(self,i,frame,det_result):
        #print(det_result)
        c = frame.shape[0]
        h = frame.shape[1]
        #print(c,h)
        if det_result.shape==(0,):
            return frame
        if type(det_result) is torch.Tensor:
            det_result = det_result.cpu().detach().numpy()
        online_targets = self.tracker.update(det_result[:, :5], frame.shape[:2], [640, 640])
        unwear_num = 0
        total_num = 0
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            #print("tid is ",tid)
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                output = detect_img([abs((tlwh[0]/640))*h, abs((tlwh[1]/640))*c, abs(((tlwh[0] + tlwh[2])/640))*h, abs(((tlwh[1] + tlwh[3]))/640)*c], frame, (0, 0, 255), str(tid))
                #plot_one_box([0,0,100,100], frame,(0, 0, 255))
                #plot_one_box([tlwh[0], tlwh[1], tlwh[2], tlwh[3]], frame, (0, 0, 255), str(tid))
                #plot_one_box([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], frame, (0, 0, 255), str(tid))
                # content = str(i) + ',' + str(tid) + ',' + str((abs(tlwh[0]/640))*h) + ',' + str((tlwh[1]/640)*c) + ',' + str(
                #     ((tlwh[2]) / 640) * h) + ',' + str(((tlwh[3])/640)*c) + ',' + '0' + ',' + '0' + ',' + '0' + '\n'
                # #print(content)
                # filename = 'car_output.txt'
                # # 使用'w'模式打开文件，如果文件不存在则创建
                # with open(filename, 'a') as file:
                #     # 写入内容
                #     file.write(content)
                if(output=="unwear"):
                    unwear_num+=1
                total_num+=1
        content = "unwear num: "+str(unwear_num)
        print(int(frame.shape[1]*2/3)) #720
        print(frame.shape[0]-10) #1910
        cv2.putText(frame, content, (int(frame.shape[1]*2/3)+20, frame.shape[0]-10), 0, 0.5 , [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        return frame,unwear_num,total_num

class yolov7(base_model):
    
    def post_processing(self, result, im=None, img=None):
        if self.model_path.endswith('pt'):
            result = non_max_suppression(result, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        elif self.model_path.endswith('onnx'):
            result = result[:, 1:]
            ratio, dwdh = letterbox(img, new_shape=tuple(self.imgsz), auto=False)[1:]
            result[:, :4] -= np.array(dwdh * 2)
            result[:, :4] /= ratio
            result[:, [4, 5]] = result[:, [5, 4]] # xyxy, cls, conf => xyxy, conf, cls
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])

        return img, result

class yolov5(base_model):
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs):
        super().__init__(model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs)
    
    def post_processing(self, result, im=None, img=None):
        if self.model_path.endswith('pt'):
            result = non_max_suppression(result, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        elif self.model_path.endswith('onnx'):
            result = non_max_suppression(torch.from_numpy(result), conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        
        return img, result

class yolov8:
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs) -> None:
        print(model_path)
        model = YOLO(model_path)
        model.info()
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        self.__dict__.update(locals())
    
    def __call__(self, data):
        if type(data) is str:
            image = cv2.imdecode(np.fromfile(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = data
        
        result = next(self.model.predict(source=image, stream=True, iou=self.iou_thres, conf=self.conf_thres, imgsz=self.imgsz, save=False, device=self.device))
        result = result.boxes.data.cpu().detach().numpy()
        for *xyxy, conf, cls in result:
            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)])
        
        return image, result
    
    def track_init(self, track_type):
        from track_utils.byte_tracker import BYTETracker, BaseTrack
        if track_type == 'ByteTrack':
            self.track_opt = ByteTrack_opt()
            self.tracker = BYTETracker(self.track_opt, frame_rate=self.track_opt.fps)
            BaseTrack._count = 0
    
    def track_processing(self, i,  frame, det_result):
        if type(det_result) is torch.Tensor:
            det_result = det_result.cpu().detach().numpy()
        online_targets = self.tracker.update(det_result[:, :5], frame.shape[:2], [640, 640])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                plot_one_box([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], frame, (0, 0, 255), str(tid))
        return frame

class rtdetr:
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs) -> None:
        model = RTDETR(model_path)
        model.info()
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        self.__dict__.update(locals())
    
    def __call__(self, data):
        if type(data) is str:
            image = cv2.imdecode(np.fromfile(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = data
        
        result = next(self.model.predict(source=image, stream=True, iou=self.iou_thres, conf=self.conf_thres, imgsz=self.imgsz, save=False, device=self.device))
        result = result.boxes.data.cpu().detach().numpy()
        for *xyxy, conf, cls in result:
            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)])
        
        return image, result
    
    def track_init(self, track_type):
        from track_utils.byte_tracker import BYTETracker, BaseTrack
        if track_type == 'ByteTrack':
            self.track_opt = ByteTrack_opt()
            self.tracker = BYTETracker(self.track_opt, frame_rate=self.track_opt.fps)
            BaseTrack._count = 0
    
    def track_processing(self, frame, det_result):
        if type(det_result) is torch.Tensor:
            det_result = det_result.cpu().detach().numpy()
        online_targets = self.tracker.update(det_result[:, :5], frame.shape[:2], [640, 640])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                plot_one_box([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], frame, (0, 0, 255), str(tid))
        return frame

def test_yolov5_track1(path):
    from track_utils.byte_tracker import BYTETracker
    # read cfg
    with open('F:/pycharmproject/segment-anything/yolo-pyqt/yolov5s.yaml') as f:
         cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # # print cfg
    print(cfg)
    # # init
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = "yolov5_output.mp4"
    fps = 25

    frame_size = (1920,1080)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    yolo = yolov5(**cfg)
    yolo.track_init('ByteTrack')
    #cap = cv2.VideoCapture("11.mp4")
    cap = cv2.VideoCapture(path)
    i = 1
    Model = model_init()
    unwear_num=0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame_size = frame.size

        #frame = frame.transpose(1,0,2)
        #print(frame.shape)
        image, result = onnx_forward1(Model,frame.copy())
        #print(result)
        image,out_num,total_num = yolo.track_processing(i, frame.copy(), result)
        if(out_num!=0):
            unwear_num+=1
        #print(type(result))
        #print(result)
        # print(image.shape)
        startppoint = int(image.shape[1]/2)
        #print(startppoint)
        content2 = str(total_num) + " people on board"
        cv2.putText(image, content2, (startppoint-300, image.shape[0]-10), 0, 2, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
        if(unwear_num>=100):
            print("Warning!!!!!!")
            content1 = "Warning!"
            cv2.putText(image, content1, (10,10), 0, 0.5, [0, 0, 255], thickness=1,lineType=cv2.LINE_AA)
            if(out_num==0):
                unwear_num-=30
        cv2.putText(image, str(i), (1800, 200), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, str(i), (image.shape[1]-30, 20), 0, 1, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('pic', 640, 640)
      # 在 'pic' 窗口中显示图像
        #print(image.shape)
        video_writer.write(image)
        cv2.imshow('pic', image)
        #print(image.shape)
        #image.transpose((2, 0, 1))
        cv2.waitKey(1)
        i += 1
    cap.release()
    video_writer.release()


def test_yolov5_track2(frame):
    from track_utils.byte_tracker import BYTETracker
    # read cfg
    with open('F:/pycharmproject/segment-anything/yolo-pyqt/yolov5s.yaml') as f:
         cfg = yaml.load(f, Loader=yaml.SafeLoader)
    yolo = yolov5(**cfg)
    yolo.track_init('ByteTrack')
    Model = model_init()
    unwear_num=0
    image, result = onnx_forward1(Model,frame.copy())
    #print(result)
    image,out_num,total_num = yolo.track_processing(frame.copy(), result)
    if(out_num!=0):
        unwear_num+=1
    #print(type(result))
    #print(result)
    # print(image.shape)
    startppoint = int(image.shape[1]/2)
    #print(startppoint)
    content2 = str(total_num) + " people on board"
    cv2.putText(image, content2, (startppoint-300, image.shape[0]-10), 0, 2, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
    if(unwear_num>=100):
        print("Warning!!!!!!")
        content1 = "Warning!"
        cv2.putText(image, content1, (10,10), 0, 0.5, [0, 0, 255], thickness=1,lineType=cv2.LINE_AA)
        if(out_num==0):
            unwear_num-=30
    # cv2.putText(image, str(i), (1800, 200), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
    # cv2.putText(image, str(i), (image.shape[1]-30, 20), 0, 1, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
    # cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('pic', 640, 640)
    return image

if __name__ == "__main__":
    test_yolov5_track1(0)


