#!/bin/bash
set -e

echo "**Downloading encoder"
wget "https://cdn.matix-media.net/dd/e46f19f4" -q --show-progress -O encoder/mars-small128.pb

echo "Downloading Yolov4 weights"
wget "https://cdn.matix-media.net/dd/a553b6eb" -q --show-progress -O yolov4/yolov4.weights

echo "Done"