## Tracking objects with Deepsort and YoloV4

Okay the tracking by ROI does not work. It seems like I need object detectors that provide prediction prediodically. And the counting number of person idea is too trivial, so it is better to try a more unique idea. 

Here is the new approach. It will be a project that track a certain object, plotting its total travelling distance overtime (normal plot), mesuring current speed (bar chart), and crop the object to a side window. 

**Features needed:**
  - The user select object to track using ROI
  - Plotting travelling distance over time as side window (normal plot)
  - Plotting average speed over time as side window (bar chart, it should plott different color for acclerating, medium speed, and walking)
  - Plotting the total amount of acclerating, walking, and medium speed
  - Cropping the object and move it to a side window

**Example:** Tracking CR7 from a football math footage

### TO-DO-LIST
  - [x] Filter tracking object with ROI contour
  - [x] Crop the object box and append it to side
  - [ ] Save object central coordinate
  - [x] Update yolov4 weight download with following link: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
  - Unitest with pytest


### Set up environment
```bash
$ conda env create -f environment.yml
$ conda activate deepsort_track 
```

### Run from terminal
```
usage: use "python tracking.py --help" for more information

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to video file
  -o OBJECTS [OBJECTS ...], --objects OBJECTS [OBJECTS ...]
                        List of tracked objects
  -c CONF, --conf CONF  Confident threshold (deafult=0.3)
  -n NMS, --nms NMS     NMS threshold (default=0.4)
  -d MCD, --mcd MCD     Max cosin distance (default=0.5)
  -f FREQ, --freq FREQ  Detection update frequency in second (default=2.0)
  -cl COLORS [COLORS ...], --colors COLORS [COLORS ...]
                        List of colors
  -s SAVE, --save SAVE  Save output video (True/False)
```

```bash
python tracking.py -v test_videos/test_video_01.mp4 -o car
```