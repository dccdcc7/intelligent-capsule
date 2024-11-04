import os, random, yaml, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random.seed(0)
import cv2
import torch
import numpy as np
from yolo_utils.utils import non_max_suppression, letterbox, scale_coords, plot_one_box
from ultralytics import YOLO, RTDETR
import torchvision
def ByteTrack_opt():
    parser = argparse.ArgumentParser("ByteTrack Param.")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
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
            result = self.model(im)
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
    
    def track_processing(self,i,frame, det_result):
        if type(det_result) is torch.Tensor:
            det_result = det_result.cpu().detach().numpy()
        online_targets = self.tracker.update(det_result[:, :5], frame.shape[:2], [640, 640])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                plot_one_box([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], frame, (0, 0, 255), str(tid))
                content = str(i)+','+str(tid)+','+str(tlwh[0])+','+str(tlwh[1])+','+str(tlwh[0] + tlwh[2])+','+str(tlwh[1] + tlwh[3])+','+'0'+','+'0'+','+'0'+'\n'
                filename = 'car_output.txt'
                # 使用'w'模式打开文件，如果文件不存在则创建
                with open(filename, 'a') as file:
                    # 写入内容
                    file.write(content)
                #print(i,tid,tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])
        return frame

class yolov7(base_model):
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
    
    def track_processing(self, i,frame, det_result):
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

def test_yolov7():
    # read cfg
    with open('yolov7-tiny.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov7(**cfg)
    image_path = '1.jpg'
    # inference
    image, _ = yolo(image_path)
    cv2.imshow('pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_yolov5():
    # read cfg
    # with open('yolov5s.yaml') as f:
    #     cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # # print cfg
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = 'cpu'
    print(device)
    weights = 'yolov5s.pt'
    datapath = 'yolov5-7.0/data/coco128.yaml'
    model = DetectMultiBackend(weights, device=device, dnn=False, data=datapath, fp16=False)
        # init
    # yolo = yolov5(**cfg)
    image_path = '2.jpg'
    # inference
    image = cv2.imread(image_path)
    image.resize(3,640,640)
    #image = torch.from_numpy(image).to(model.device)
    if len(image.shape) == 3:
        image = image[None]  # expand for batch dim
    image = torch.from_numpy(image).to(torch.float)
    print(type(image))
    print(image)
    image = image.permute(0,1,3,2)
    print(image.shape)
    image = image.to(device)
    image_out = model(image)
    print(type(image_out))
    # numpy_list = []
    # for tensor in image_out:
    #     # 将张量移动到CPU
    #     for tensor1 in tensor:
    #         # 将张量移动到CPU
    #         tensor_cpu = tensor1.cpu()
    #         # 将张量转换为NumPy数组
    #         numpy_array = tensor_cpu.numpy()
    #         # 将转换后的数组添加到新列表中
    #         numpy_list.append(numpy_array)
    # print(numpy_list)
    # print(type(numpy_list))
    # numpy_list = np.array(numpy_list,dtype=object)
    # print(type(numpy_list))
    # cv2.imshow('pic', numpy_list)
    # cv2.imwrite("./11.jpg",numpy_list)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

def test_yolov5_track():
    from track_utils.byte_tracker import BYTETracker
    # read cfg
    with open('yolov5s.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov5(**cfg)
    yolo.track_init('ByteTrack')
    
    cap = cv2.VideoCapture('3.mp4')
    i=1
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        print(type(frame))
        image, result = yolo(frame.copy())
        print(result)
        print(type(result))
        image = yolo.track_processing(i,frame.copy(), result)
        
        cv2.imshow('pic', image)
        cv2.waitKey(20)
        i=i+1

def test_yolov7_track():
    from track_utils.byte_tracker import BYTETracker
    # read cfg
    with open('yolov7-tiny.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov7(**cfg)
    yolo.track_init('ByteTrack')
    
    cap = cv2.VideoCapture('1.mp4')
    
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        
        image, result = yolo(frame.copy())
        image = yolo.track_processing(frame.copy(), result)
        
        cv2.imshow('pic', image)
        cv2.waitKey(20)

if __name__ == '__main__':
    #test_yolov5()
    # test_yolov7()
    test_yolov5_track()
    #test_yolov7_track()