import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from helper import Helper, argparse_init
args = argparse_init()
from deep_sort import nn_matching
import cv2
import numpy as np
import traceback
from tools import generate_detections as gdet
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from collections import deque
from Yolov4 import Yolo
import logging


# Create log history
FILE_NAME = os.path.basename(__file__).split('.')[0]
logging.basicConfig(filename="Logs/%s.log"%(FILE_NAME),
                    filemode="w", 
                    level=logging.INFO, 
                    format="%(asctime)s %(message)s")


# Create a queue to store bounding box center
pts = [deque(maxlen=30) for _ in range(9999)]

# Initilaize feature encoder for DeepSort
encoder_path = os.path.join("encoder","mars-small128.pb")
encoder = gdet.create_box_encoder(encoder_path,batch_size=1)

# Tracking parameters
metric = nn_matching.NearestNeighborDistanceMetric("cosine", args["mcd"], None)
tracker = Tracker(metric)

if __name__=="__main__":

    video_path = args["video"]
    assert os.path.isfile(video_path), "Could not find %s"%video_path

    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()

    if args["mode"]:
        helper = Helper(objects=args["objects"], colors=args["colors"])        
        roi = cv2.selectROI(frame,fromCenter=False, showCrosshair=False)
        bbox = [int(x) for x in roi]
        logging.info("box manually: %s"%bbox)

        class_names = args["objects"]
        boxes = [bbox]

    else:
    # Initialize yolo
        yolo = Yolo(conf_thresh=args["conf"], 
                    nms_thresh=args["nms"], 
                    detecting_objs=args["objects"])
        helper = Helper(objects=yolo.detecting_objs, colors=args["colors"])        

    while ret:    
        try:
            ret, frame = cap.read()
            drawed_frame = frame.copy()
            
            if not args["mode"]:
                boxes, class_names = yolo.detect_image(frame)
                logging.info("boxes by yolo: %s"%boxes[0])
                # features = encoder(frame,boxes)
                # logging.info("features yolov4: %s"%features)
                # logging.info("Features: %s"%features)

            features = encoder(frame, boxes)
            # Score to 1.0 here
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]


        ## Call the tracker
            tracker.predict()
            tracker.update(detections)
            
            # indexIDs = []
            c = []
            boxes=[]

            for class_name, det in zip(class_names, detections):
                bbox = det.to_tlbr()

                # plot bounding box with bounding_box package
                helper.drawing_bbox(drawed_frame, bbox, class_name)

                # plot bounding box with OpenCV
                # cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            
            for index, track in enumerate(tracker.tracks):
                if not track.is_confirmed() or track.time_since_update > args["freq"] or index >= len(class_names):
                    continue
                
                bbox = track.to_tlbr()
                boxes.append(bbox)
                # logging.info("boxes.append(bbox) %s"%boxes)

                helper.drawing_bbox(drawed_frame, 
                                    bbox, 
                                    class_name=class_names[index], 
                                    text_id="id: %s"%track.track_id)
                
                # bbox_center_point(x,y)
                center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
                
                # track_id[center]
                pts[track.track_id].append(center)
                thickness = 5
                
                # center point 
                cv2.circle(drawed_frame, (center), 1, (0,0,255), thickness)
                
                # draw motion path
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64/float(j+1))*2)
                    cv2.line(drawed_frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(0,0,255),thickness)
                
            # cv2.putText(drawed_frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
            # cv2.putText(drawed_frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
            cv2.imshow('Deep_SORT', drawed_frame)
            
            
            key=cv2.waitKey(1) & 0xff
            if key==27:
                break

        except Exception as e:
            print(traceback.format_exc())

    
    cap.release()

