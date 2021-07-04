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