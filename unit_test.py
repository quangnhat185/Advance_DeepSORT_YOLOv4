import os


def test_enoder_path():
    encoder_path = os.path.join("encoder","mars-small128.pb")
    assert os.path.isfile(encoder_path), "Could not find %s"%encoder_path
    
def test_yolov4_path():
    YOLOV4_FOLDER = "yolov4"
    label_path = os.path.join(YOLOV4_FOLDER,"coco.names")
    weight_path = os.path.join(YOLOV4_FOLDER, "yolov4.weights")
    config_path = os.path.join(YOLOV4_FOLDER, "yolov4.cfg")
    assert os.path.isfile(label_path), "Could not find %s"%label_path
    assert os.path.isfile(config_path), "Could not find %s"%config_path
    assert os.path.isfile(weight_path), "Could not find %s, make sure you already downloaded yolov4 weight"%weight_path