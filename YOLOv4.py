import cv2
import numpy as np
import os

YOLOV4_FOLDER = "yolov4"
LABELS_PATH = os.path.join(YOLOV4_FOLDER,"coco.names")
WEIGHT_PATH = os.path.join(YOLOV4_FOLDER, "yolov4.weights")
CONFIG_PATH = os.path.join(YOLOV4_FOLDER, "yolov4.cfg")

class YOLO(object):
    def __init__(self, conf_thresh=0.3, nms_thresh=0.4, detecting_objs=["car", "truck"]):

        self.weight_path = WEIGHT_PATH
        self.config_path = CONFIG_PATH
        self.label_path = LABELS_PATH
        self.labels = None
        self.detecting_objs = detecting_objs
        self._colors = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray', 'silver']

        # Unit tests
        assert os.path.isfile(self.label_path), "Could not find %s"%self.label_path
        assert os.path.isfile(self.config_path), "Could not find %s"%self.config_path
        assert os.path.isfile(self.weight_path), "Could not find %s"%self.weight_path

        # Load coco labels
        with open(LABELS_PATH) as file:
            self.labels = file.read().strip().split("\n")        

        self.conf_threshold = conf_thresh
        self.nms_thesh = nms_thresh
        self.size=(416,416)

        net = cv2.dnn.readNetFromDarknet(self.config_path, self.weight_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416,416), scale=1/255, swapRB=True)
        print("[INFO] Created YoloV4 model successfully")

    def detect_image(self, image):

        classes, scores, boxes = self.model.detect(image, self.conf_threshold, self.nms_thesh)

        return_boxes = []
        return_class_names=[]

        for class_idx, box in zip(classes,boxes):
            class_name = self.labels[int(class_idx)]
            if class_name in self.detecting_objs:
                return_boxes.append(box)
                return_class_names.append(class_name)
        
        return return_boxes, return_class_names

    def drawing_boxes(self, frame, boxes, class_names, colors=[]):
        dict_color = dict

        if not colors:
            rand_colors = np.random.choice(self._colors, len(self.detecting_objs))
            dict_color = dict(zip(self.detecting_objs, rand_colors))
        else:
            assert len(colors) == len(self.detecting_objs), "The number of colors need to match the number of tracking objects: %i color(s) but %i tracking object[s]"%(len(colors), len(self.detecting_objs))

            dict_color = dict(zip(self.detecting_objs, colors))

        return_frame = frame.copy()
        for name, box in zip(class_names,boxes):
            (x,y) = (box[0],box[1])
            (w,h) = (box[2],box[3])
            bb.add(return_frame,x,y,x+w,y+h,name,str(dict_color[name]))

        return return_frame



if __name__=="__main__":
    yolo = YOLO(detecting_objs=["car","truck"])
    
    cap = cv2.VideoCapture("test_video.mp4")
    ret, frame = cap.read()

    while ret:    
        ret, frame = cap.read()
        boxes, class_names = yolo.detect_image(frame)
        drawed_frame = yolo.drawing_boxes(frame, boxes, class_names,colors=["green","red"])
        
        cv2.imshow("predicted", drawed_frame)
        
        key=cv2.waitKey(1) & 0xff
        if key==27:
            break



