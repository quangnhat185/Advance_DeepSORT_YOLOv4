import argparse
import textwrap
# Argparse
parser = argparse.ArgumentParser(
    usage='use "python %(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter
    )
parser.add_argument("-v","--video",
                    type=str, 
                    required=True, 
                    help="Path to video file")
parser.add_argument("-o","--object", 
                    required=True, 
                    nargs="+", 
                    help="List of tracked objects")
parser.add_argument("-c", "--conf", 
                    type=float,
                    required=False, 
                    default=0.3, 
                    help="Confident threshold (deafult=0.3)" )
parser.add_argument("-n", "--nms", 
                    type=float,
                    required=False, 
                    default=0.4, 
                    help="NMS threshold (default=0.4)")
parser.add_argument("-d", "--mcd",
                    type=float,
                    required = False,
                    default=0.5,
                    help="Max cosin distance (default=0.5)"
                    )
parser.add_argument("-f","--freq",
                    type=float,
                    required=False,
                    default = 1.0,
                    help="Detection update frequency in second (default=1.0)"
                    )
parser.add_argument("-m", "--mode",
                    type=int,
                    required=False, 
                    default=0, 
                    help=textwrap.dedent('''\
                    0: Tracking object by class name (default)
                    1: Tracking object by ID'''))

args = vars(parser.parse_args())


import os
import cv2
import numpy as np
from tools import generate_detections as gdet
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from collections import deque
from YOLOv4 import YOLO
from datetime import datetime
import logging

# Generate log files
FILE_NAME = os.path.basename(__file__)
date_time = datetime.now().strftime("%d_%m_%Y")
logging.basicConfig(filename="Logs/%s_[%s].log"%(FILE_NAME,date_time),
                    filemode="w", 
                    level=logging.INFO, 
                    format="%(asctime)s %(message)s")




#Definition of the parameters
max_cosin_distance = 0.5
nn_budget = None

counter = []

pts = [deque(maxlen=30) for _ in range(9999)]

encoder_path = os.path.join("encoder","mars-small128.pb")
assert os.path.isfile(encoder_path), "Could not find %s"%encoder_path
encoder = gdet.create_box_encoder(encoder_path,batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosin_distance, nn_budget)
tracker = Tracker(metric)

if __name__=="__main__":
    # yolo = YOLO(detecting_objs=["car","truck"])
    yolo = YOLO(detecting_objs=["person"])
    np.random.seed(100)
    dict_color = dict
    COLORS = np.random.randint(0,255,size=(200,3), dtype="uint8")   

    video_path = os.path.join("test_videos", "test_video_01.mp4")
    assert os.path.isfile(video_path), "Could not find %s"%video_path

    # cap = cv2.VideoCapture("test_videos/test_video_01.mp4")
    cap = cv2.VideoCapture("test_videos/test_video_02.mp4")
    
    ret, frame = cap.read()

    while ret:    
        ret, frame = cap.read()
        boxes, class_names = yolo.detect_image(frame,)

        features = encoder(frame,boxes)
        logging.info("Features: %s"%features)

        # Score to 1.0 here
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

       ## Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        i = int(0)
        indexIDs = []
        c = []
        
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            
        for index, track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 2.5:
                continue

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)

            cv2.putText(frame, str(class_names[index]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
            
            
            i+=1
            # bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            
            # center point 
            cv2.circle(frame, (center), 1, color, thickness)
            
            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64/float(j+1))*2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
            
        count = len(set(counter))  
        cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.imshow('Deep_SORT', frame)
        
        
        key=cv2.waitKey(1) & 0xff
        if key==27:
            break
    
    cap.release()

