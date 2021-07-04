## Tracking objects with Deepsort and YoloV4


### TO-DO-LIST
- [ ] Write bounding boxes drawing functions in [helper.py](./helper.py) and deloy it in the main function
- [ ] Fix the colors issue. The idea is there each object will be assigned to only once color
- [ ] [tracking.py](./tracking.py): There should be two modes: tracking all pre-indicate objects or tracking an object following its id
- [ ] [counting.py](./counting.py) Drawing a line and count the number of persons get in and get our the line
- [ ] The running scripts should contains parsing arguments as following:
  - Video path -> String
  - Tracking mode (if available) -> 0 , 1
  - Threshold -> float
  - Pre-indicate objects -> List[String]
  
- Try to understand the hyperparmeter: confident threshold, nsm, max cosindistance, mode


```json
usage: use "python tracking.py --help" for more information

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to video file
  -o OBJECT [OBJECT ...], --object OBJECT [OBJECT ...]
                        List of tracked objects
  -c CONF, --conf CONF  Confident threshold (deafult=0.3)
  -n NMS, --nms NMS     NMS threshold (default=0.4)
  -d MCD, --mcd MCD     Max cosin distance (default=0.5)
  -f FREQ, --freq FREQ  Detection update frequency in second (default=1.0)
  -m MODE, --mode MODE  0: Tracking object by class name (default)
                        1: Tracking object by ID
```