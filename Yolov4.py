import cv2
import numpy as np
import os
from helper import Helper

YOLOV4_FOLDER = "yolov4"
LABELS_PATH = os.path.join(YOLOV4_FOLDER,"coco.names")
WEIGHT_PATH = os.path.join(YOLOV4_FOLDER, "yolov4.weights")
CONFIG_PATH = os.path.join(YOLOV4_FOLDER, "yolov4.cfg")

class Yolo(object):
    def __init__(self, conf_thresh, nms_thresh, detecting_objs=[]):

        self.weight_path = WEIGHT_PATH
        self.config_path = CONFIG_PATH
        self.label_path = LABELS_PATH
        self.labels = None
        self.detecting_objs = detecting_objs

        # Load coco labels
        with open(LABELS_PATH) as file:
            self.labels = file.read().strip().split("\n")       

        for class_name in self.detecting_objs:
            assert class_name in self.labels, "This object class does not exist. Please select object classes from the following list:\n\n %s"%(self.labels) 

        self.conf_threshold = conf_thresh
        self.nms_thesh = nms_thresh
        self.size=(416,416)

        net = cv2.dnn.readNetFromDarknet(self.config_path, self.weight_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416,416), scale=1/255, swapRB=True)
        print("[INFO] YoloV4 is successfully initialized!")

    def to_tlbr(self, bbox):
        """Convert bounding box coordinate format to `[top left, bottom right]` format

        Parameters
        ----------
        bbox : ndarray
            bounding box in default yolov4 format

        Returns
        -------
        ret : ndarray
            bounding box in `[top left, bottom right]` format
        """
        
        tlwh = np.asarray(bbox, dtype=np.float)
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def detect_image(self, image):
        """Return object's bounding box and its class name

        Parameters
        ----------
        image : ndarray
            Representation of an image in `ndarray` format

        Returns
        -------
        return_boxes : List of array
            List of bounding box coordinates

        return_class_names : List of string
            List of object's class name
        """
        classes, scores, boxes = self.model.detect(image, self.conf_threshold, self.nms_thesh)

        return_boxes = []
        return_class_names=[]

        for class_idx, box in zip(classes,boxes):
            class_name = self.labels[int(class_idx)]
            if class_name in self.detecting_objs and box[2]<500:
                return_boxes.append(box)
                return_class_names.append(class_name)
        
        return return_boxes, return_class_names


if __name__=="__main__":
    yolo = Yolo(conf_thresh=0.3, nms_thresh=0.4,detecting_objs=["car","truck"])
    helper = Helper(objects=yolo.detecting_objs, colors=["green", "blue"])
    cap = cv2.VideoCapture("test_videos/test_video_01.mp4")
    ret, frame = cap.read()

    while ret:    
        ret, frame = cap.read()
        boxes, class_names = yolo.detect_image(frame)
        drawed_frame = frame.copy()

        for class_name, bbox in zip(class_names, boxes):
            bbox = yolo.to_tlbr(bbox)
            helper.drawing_bbox(drawed_frame, bbox, class_name)

        cv2.imshow("predicted", drawed_frame)
        
        np.linspace
        key=cv2.waitKey(1) & 0xff
        if key==27:
            break



