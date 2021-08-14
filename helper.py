import numpy as np
import argparse
import cv2
import textwrap
import logging
from bounding_box import bounding_box as bb



# Argparse
def argparse_init():
    parser = argparse.ArgumentParser(
        usage='use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter
        )
    parser.add_argument("-v","--video",
                        type=str, 
                        required=True, 
                        help="Path to video file")
    parser.add_argument("-o","--objects", 
                        required=True, 
                        nargs="+", 
                        default=[],
                        help="List of tracked objects")
    parser.add_argument("-c", "--conf", 
                        type=float,
                        required=False, 
                        default=0.3, 
                        help="Confident threshold (deafult=0.3)" )
    parser.add_argument("-n", "--nms", 
                        type=float,
                        required=False, 
                        default=0.4, 
                        help="NMS threshold (default=0.4)")
    parser.add_argument("-d", "--mcd",
                        type=float,
                        required = False,
                        default=0.5,
                        help="Max cosin distance (default=0.5)"
                        )
    parser.add_argument("-f","--freq",
                        type=float,
                        required=False,
                        default = 2.0,
                        help="Detection update frequency in second (default=2.0)"
                        )
    parser.add_argument("-m", "--mode",
                        type=int,
                        required=False, 
                        default=0, 
                        help=textwrap.dedent('''\
                        0: Tracking object by class name (default)
                        1: Tracking with drag box'''))
    parser.add_argument("-cl", "--colors",
                        required=False,
                        default=[],
                        nargs="+",
                        help="List of colors"
                        )
    parser.add_argument("-s","--save",
                        type=bool,
                        default=False,
                        help="Save output video (True/False)"
                        )

    args = vars(parser.parse_args())
    return args

class Helper():
    """[summary]
    """
    def __init__(self, objects, colors=[]):
        self.COLORS = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 'orange', 'red', 'maroon', 'fuchsia', 'purple']
        self.objects = objects
        self.num_classes = len(self.objects)
        self.color_dict = {}

        if not colors:
            rand_colors = np.random.choice(self.COLORS, self.num_classes)
            self.color_dict = dict(zip(self.objects, rand_colors))
        else:
            assert len(colors) == self.num_classes, "The number of colors need to match the number of tracking objects: %i color(s) but %i tracking object[s]"%(len(colors), self.num_classes)

            for color in colors:
                assert color in self.COLORS, "This color is not supported. Please select the color from the following list:\n\n %s"%self.COLORS

            self.color_dict = dict(zip(self.objects, colors))
            
        self.color_dict["NA"] = "black"



    def drawing_bbox(self, drawed_frame, bbox, class_name=None, text_id=None, draw_with_opencv=False):
        """[summary]

        Parameters
        ----------
        drawed_frame : [type]
            [description]
        bbox : [type]
            [description]
        class_name : [type], optional
            [description], by default None
        text_id : [type], optional
            [description], by default None
        draw_with_opencv : bool, optional
            [description], by default False
        """
        
        if not draw_with_opencv:
            (x_1,y_1) = (bbox[0],bbox[1])
            (x_2,y_2) = (bbox[2],bbox[3])
            bb.add(drawed_frame,x_1,y_1,x_2,y_2,class_name,str(self.color_dict[class_name]))
            
        else:
            cv2.rectangle(drawed_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),([255,0,0]), 3)
            cv2.putText(drawed_frame, str(class_name),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,0,0),2)

        if text_id:
            cv2.putText(drawed_frame,text_id,(int(bbox[0]+15), int(bbox[1] -20)),0, 5e-3 * 120, (0,0,255),2)