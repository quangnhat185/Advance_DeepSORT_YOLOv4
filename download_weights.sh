#!/bin/bash
set -e

mkdir encoder output

echo "**Downloading encoder"
wget "https://github.com/Qidian213/deep_sort_yolov3/raw/master/model_data/mars-small128.pb" -q --show-progress -O encoder/mars-small128.pb

echo "Downloading Yolov4 weights"
wget "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" -q --show-progress -O yolov4/yolov4.weights

echo "Done"