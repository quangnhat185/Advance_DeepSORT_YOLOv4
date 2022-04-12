# Advance tracking with DeepSORT and YOLOv4
<p align="center">
<img src="https://github.com/quangnhat185/Media/blob/main/Advance_DeepSort_YoloV4/final_result.gif" width="1920" />
</p>

## Key features
  - Select single object to track with Region of Interest (ROI)
  - Isolate tracking object as cropped frame
  - Plotting travelled distance and velocity (Applicable only if camera is static)
  - Tracjectory representation

## Set up environment
```bash
$ conda env create -f environment.yml
$ conda activate deepsort_track 
```

## Run from terminal
```
usage: use "python tracking.py --help" for more information

Tracking with DeepSort

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to video file
  -t TARGET [TARGET ...], --target TARGET [TARGET ...]
                        Type of tracking target (person, car, etc)
  -c CONF, --conf CONF  Confident threshold (deafult=0.3)
  -n NMS, --nms NMS     NMS threshold (default=0.4)
  -d MCD, --mcd MCD     Max cosin distance (default=0.5)
  -f FREQ, --freq FREQ  Detection update frequency in second (default=2.0)
  -cl COLORS [COLORS ...], --colors COLORS [COLORS ...]
                        List of colors
  -s SAVE, --save SAVE  Save output video (True/False)

```

```bash
python tracking.py -v test_videos/football.mp4 -t person
```

## Citation
 ```
 @inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}

@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection}, 
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 ```
## Credit
- [Github: LinhanDai/yolov4-deepsort](https://github.com/LinhanDai/yolov4-deepsort)
- [Github: nwoje/deep_sort](https://github.com/nwojke/deep_sort)
- [Github: LeonLok/Deep-SORT_YOLOv4](https://github.com/LeonLok/Deep-SORT-YOLOv4)

