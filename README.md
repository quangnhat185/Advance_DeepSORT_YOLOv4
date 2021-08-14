## Tracking objects with Deepsort and YoloV4

Okay the tracking by ROI does not work. It seems like I need object detectors that provide prediction prediodically. And the counting number of person idea is too trivial, so it is better to try a more unique idea. 

Here is the new approach. It will be a project that track a certain object, plotting its total travelling distance overtime (normal plot), mesuring current speed (bar chart), and crop the object to a side window. 

**Features needed:**
  - Return a list of objects, the user select the object he/she wants to track by mouse (use ROI to create a contour around desired object)
  - Plotting travelling distance over time as side window (normal plot)
  - Plotting average speed over time as side window (bar chart, it should plott different color for acclerating, medium speed, and walking)
  - Plotting the total amount of acclerating, walking, and medium speed
  - Cropping the object and move it to a side window

**Example:** Tracking CR7 from a football math footage

### TO-DO-LIST
- [ ] [tracking.py](./tracking.py): There should be two modes: tracking all pre-indicate objects or tracking an object following its id
- [ ] [counting.py](./counting.py) Drawing a line and count the number of persons get in and get our the line
- [ ] Function to save video from input argument
- [x] Write bounding boxes drawing functions in [helper.py](./helper.py) and deloy it in the main function
- [x] Fix the colors issue. The idea is there each object will be assigned to only once color
- [x] The running scripts should contains parsing arguments as following:
  - Video path -> String
  - Tracking mode (if available) -> 0 , 1
  - Threshold -> float
  - Pre-indicate objects -> List[String]
  
- Try to understand the hyperparmeter: confident threshold, nsm, max cosindistance, mode


```consolec
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
  -m MODE, --mode MODE  0: Tracking object by class name (default)
                        1: Tracking object by ID number
  -cl COLORS [COLORS ...], --colors COLORS [COLORS ...]
                        List of colors
  -s SAVE, --save SAVE  Save output video (True/False)
```

```bash
python tracking.py -v test_videos/test_video_01.mp4 -o car
python tracking.py -v test_videos/test_video_01.mp4 -m 1 -o object
```