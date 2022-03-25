import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
import time
import pickle
import logging


# Create log history
FILE_NAME = os.path.basename(__file__).split(".")[0]
logging.basicConfig(
    filename="Logs/%s.log" % (FILE_NAME),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

# Create a queue to store bounding box center
pts = [deque(maxlen=30) for _ in range(9999)]

# Initilaize feature encoder for DeepSort
encoder_path = os.path.join("encoder", "mars-small128.pb")
encoder = gdet.create_box_encoder(encoder_path, batch_size=1)

# Tracking parameters
metric = nn_matching.NearestNeighborDistanceMetric("cosine", args["mcd"], None)
tracker = Tracker(metric)

if __name__ == "__main__":

    video_path = args["video"]
    assert os.path.isfile(video_path), "Could not find %s" % video_path

    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(4))
    width = int(cap.get(3))
    ret, frame = cap.read()

    yolo = Yolo(
        conf_thresh=args["conf"],
        nms_thresh=args["nms"],
        detecting_objs=args["objects"],
    )
    helper = Helper(objects=yolo.detecting_objs, colors=args["colors"])

    roi_select_status = False

    object_pts = dict()

    while ret:
        try:
            ret, frame = cap.read()

            drawed_frame = frame.copy()

            # Obtain detected bounding boxes with yolo
            boxes, class_names = yolo.detect_image(frame)
            logging.info("boxes by yolo: %s" % boxes[0])
            # features = encoder(frame,boxes)
            # logging.info("features yolov4: %s"%features)
            # logging.info("Features: %s"%features)

            # extract features from detected boxes
            features = encoder(frame, boxes)

            # Combine bounding boxes and corresponding feature into an instance
            # of Detection class
            detections = [
                Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)     
            ]
                                
            logging.info("boxes, features: {} ----- {}".format(boxes, features))

            ## Call the tracker
            tracker.predict()
            tracker.update(detections)
            tracker.tracks

            # Select tracking object with ROI
            key = cv2.waitKey(1) & 0xFF

            if not roi_select_status:
                for class_name, det in zip(class_names, detections):
                    bbox = det.to_tlbr()
                    cv2.putText(
                        drawed_frame,
                        "Press c to select track object",
                        (int(width/2)-100, height-100),
                        0,
                        5e-3 * 200,
                        (0,0,255),
                        2,
                    )                    

                    # plot bounding box
                    helper.drawing_bbox(drawed_frame, bbox, class_name)

                if key==ord("c"):
                    roi = helper.extract_roi(drawed_frame)
                    roi_select_status = True
                    logging.info("ROI: {}".format(roi))
                    
                    # extract centroids of all detected bounding boxes
                    bbox_centers = []
                    track_ids = []
                    for index, track in enumerate(tracker.tracks):
                        bbox = track.to_tlbr()
                        center = (
                            int(((bbox[0]) + (bbox[2])) / 2),
                            int(((bbox[1]) + (bbox[3])) / 2),
                        )
                        bbox_centers.append(center)
                    bbox_centers = np.array(bbox_centers)

                    # return track id based on roi and bbox centroids
                    track_id = helper.tracking_id_from_roi(roi, bbox_centers)
                    logging.info(f"bbox_centers: {bbox_centers}")
                    logging.info("track.track_id: {}".format(track.track_id))

            elif roi_select_status:
                for index, track in enumerate(tracker.tracks):
                    if (
                        not track.is_confirmed()
                        or track.track_id != track_id
                        or track.time_since_update > args["freq"]
                        or index >= len(class_names)
                    ):
                        continue
                    
                    bbox = track.to_tlbr()

                    # crop tracked object based on boundingbox
                    tracked_frame = helper.crop_tracked_frame(bbox, frame, offset=20)
                    cv2.imshow("Crop image", tracked_frame)
                    

                    logging.info("track.trackid: {}".format(track.track_id))
                    helper.drawing_bbox(
                        drawed_frame,
                        bbox,
                        class_name=class_names[index],
                        text_id="id: %s" % track.track_id,
                    )

                    # bbox_center_point(x,y)
                    center = (
                        int(((bbox[0]) + (bbox[2])) / 2),
                        int(((bbox[1]) + (bbox[3])) / 2),
                    )

                    object_pts[center] = time.time()
                    logging.info("object_pts: {}".format(str(object_pts)))

                    # center point
                    thickness = 5
                    cv2.circle(drawed_frame, (center), 1, (0, 0, 255), thickness)

            cv2.namedWindow("   ")  
            cv2.moveWindow("Deep SORT", 0,0)
            cv2.imshow("Deep SORT", drawed_frame)
            if key == 27:
                break

        except Exception as e:
            print(traceback.format_exc())

    # with open("object_pts", "wb") as f:
        # pickle.dump(object_pts, f)

    cap.release()
