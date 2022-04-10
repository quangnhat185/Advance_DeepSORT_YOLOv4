import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
from data_plotting import DataPlot
import matplotlib.pyplot as plt

import time
import pickle
import logging


# Size (width, height) of cropped frame with tracked object
CROP_FRAME_SIZE =  (180,320)
PLOT_FRAME_SIZE = (1386,716)

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
    
    # configure video writers
    if args["save"]:
        number_frame = 30
        video_size = (width, height)
        writer_drawed_frame = cv2.VideoWriter("./output/tracked_video.mp4",cv2.VideoWriter_fourcc(*'MP4V'),
        number_frame,video_size)
        writer_drawed_blank = cv2.VideoWriter("./output/path_follower.mp4",cv2.VideoWriter_fourcc(*'MP4V'),
        number_frame,video_size)
        writer_drawed_cropped = cv2.VideoWriter("./output/cropped_frame.mp4",cv2.VideoWriter_fourcc(*'MP4V'),
        number_frame,CROP_FRAME_SIZE)
        writer_drawed_plot = cv2.VideoWriter("./output/plotted_frame.mp4",cv2.VideoWriter_fourcc(*'MP4V'),
        number_frame,PLOT_FRAME_SIZE)
        
    # initialize YoloV4 detector
    yolo = Yolo(
        conf_thresh=args["conf"],
        nms_thresh=args["nms"],
        detecting_objs=args["target"],
    )
    helper = Helper(objects=yolo.detecting_objs, colors=args["colors"])

    roi_select_status = False

    object_pts = dict()

    plotter = DataPlot(save=args["save"])

    blank = np.zeros_like(frame)

    while ret:
        try:    
            ret, frame = cap.read()

            drawed_frame = frame.copy()
            drawed_blank = blank.copy()

            # Obtain detected bounding boxes with yolo
            boxes, class_names = yolo.detect_image(frame)
            logging.info("boxes by yolo: %s" % boxes[0])

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
                    
                    # extract centroids of all detected bounding boxes
                    bbox_centers = []
                    track_ids = []
                    for index, track in enumerate(tracker.tracks):
                        bbox = track.to_tlbr()
                        center = helper.bbox_center(bbox, is_tlbr=True)
                        bbox_centers.append(center)
                    bbox_centers = np.array(bbox_centers)

                    # return track id based on roi and bbox centroids
                    track_id = helper.tracking_id_from_roi(roi, bbox_centers)

            # track only selected target within roi
            elif roi_select_status:
                for index, track in enumerate(tracker.tracks):
                    track_log = f"track.is_confirmed: {track.is_confirmed()} \n" + f"track.track_id != track_id: {track.track_id != track_id}"
                    
                    if (
                        not track.is_confirmed()
                        or track.track_id != track_id
                        or track.time_since_update > args["freq"]
                    ):  
                        continue
                    
                    bbox = track.to_tlbr()

                    # crop tracked object based on boundingbox
                    cropped_frame = helper.crop_tracked_frame(bbox, frame, offset=20)
                    # print(cropped_frame.shape)
                    cv2.imshow("Crop image", cropped_frame)
                    cropped_frame = cv2.resize(cropped_frame, CROP_FRAME_SIZE)
                    writer_drawed_cropped.write(cropped_frame)

                    # draw bounding box around target
                    helper.drawing_bbox(
                        drawed_frame,
                        bbox,
                        class_name=class_names[index],
                        text_id="id: %s" % track.track_id,
                    )

                    # saving center of the bounding box
                    center = helper.bbox_center(bbox, is_tlbr=True)
                    object_pts[center] = time.time()
                    

                    # Plot live data
                    if len(object_pts)>1:
                        plotter.plot(object_pts, pause=0.5)
                        plot_frame = cv2.imread("./output/plot/tracking_plot.png"),
                        plot_frame = cv2.resize(plot_frame[0], (PLOT_FRAME_SIZE))
                        writer_drawed_plot.write(plot_frame)

                    # Draw center points
                    thickness = -1
                    cv2.circle(drawed_frame, (center), 5, (0, 0, 255), thickness)
                    cv2.circle(drawed_blank, (center), 15, (0,0,255), thickness)

                    
                    # Draw motion path
                    num_of_draw_pts = len(object_pts) if len(object_pts) < 20 else 20   
                    for j in range(1,num_of_draw_pts):
                        cv2.line(drawed_blank,(list(object_pts.keys())[-j]), (list(object_pts.keys())[-j-1]),(0,255,0),thickness=4)                    

            # Show frame
            cv2.namedWindow("   ")  
            cv2.moveWindow("Deep SORT", 0,0)
            cv2.imshow("Deep SORT", drawed_frame)
            cv2.imshow("Background view", drawed_blank)

            # pres ESC to break the loop
            if key == 27:
                break
            
            # Wrtie frames as video
            if args["save"] and roi_select_status:
                writer_drawed_frame.write(drawed_frame)
                writer_drawed_blank.write(drawed_blank)

        except Exception as e:
            print(traceback.format_exc())

    # with open("object_pts", "wb") as f:
    #     pickle.dump(object_pts, f)

    writer_drawed_frame.release()
    writer_drawed_blank.release()
    writer_drawed_cropped.release()
    writer_drawed_plot.release()
    cap.release()
