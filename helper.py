import numpy as np
import argparse
import cv2
import textwrap
import logging
from bounding_box import bounding_box as bb


# Argparse
def argparse_init():
    parser = argparse.ArgumentParser(
        description = "Tracking with DeepSort",
        usage='use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v", "--video", type=str, required=True, help="Path to video file"
    )
    parser.add_argument(
        "-o",
        "--objects",
        required=True,
        nargs="+",
        default=[],
        help="List of tracked objects",
    )
    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        required=False,
        default=0.3,
        help="Confident threshold (deafult=0.3)",
    )
    parser.add_argument(
        "-n",
        "--nms",
        type=float,
        required=False,
        default=0.4,
        help="NMS threshold (default=0.4)",
    )
    parser.add_argument(
        "-d",
        "--mcd",
        type=float,
        required=False,
        default=0.5,
        help="Max cosin distance (default=0.5)",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=float,
        required=False,
        default=2.0,
        help="Detection update frequency in second (default=2.0)",
    )

    parser.add_argument(
        "-cl", "--colors", required=False, default=[], nargs="+", help="List of colors"
    )
    parser.add_argument(
        "-s", "--save", type=bool, default=False, help="Save output video (True/False)"
    )

    args = vars(parser.parse_args())
    return args


class Helper:
    def __init__(self, objects, colors=[]):
        self.COLORS = [
            "navy",
            "blue",
            "aqua",
            "teal",
            "olive",
            "green",
            "lime",
            "yellow",
            "orange",
            "red",
            "maroon",
            "fuchsia",
            "purple",
        ]
        self.objects = objects
        self.num_classes = len(self.objects)
        self.color_dict = {}

        if not colors:
            rand_colors = np.random.choice(self.COLORS, self.num_classes)
            self.color_dict = dict(zip(self.objects, rand_colors))
        else:
            assert len(colors) == self.num_classes, (
                "The number of colors need to match the number of tracking objects: %i color(s) but %i tracking object[s]"
                % (len(colors), self.num_classes)
            )

            for color in colors:
                assert color in self.COLORS, (
                    "This color is not supported. Please select the color from the following list:\n\n %s"
                    % self.COLORS
                )

            self.color_dict = dict(zip(self.objects, colors))

        self.color_dict["NA"] = "black"

    def drawing_bbox(
        self, drawed_frame, bbox, class_name="NA", text_id=None, draw_with_opencv=False
    ):
        """Draw bbox on image

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
            (x_1, y_1) = (bbox[0], bbox[1])
            (x_2, y_2) = (bbox[2], bbox[3])
            bb.add(
                drawed_frame,
                x_1,
                y_1,
                x_2,
                y_2,
                class_name,
                str(self.color_dict[class_name]),
            )

        else:
            cv2.rectangle(
                drawed_frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                ([255, 0, 0]),
                3,
            )
            cv2.putText(
                drawed_frame,
                str(class_name),
                (int(bbox[0]), int(bbox[1] - 20)),
                0,
                5e-3 * 150,
                (255, 0, 0),
                2,
            )

        if text_id:
            cv2.putText(
                drawed_frame,
                text_id,
                (int(bbox[0] + 15), int(bbox[1] - 20)),
                0,
                5e-3 * 120,
                (0, 0, 255),
                2,
            )

    @classmethod
    def extract_roi(cls, frame):
        roi = cv2.selectROI("ROI selector", frame, fromCenter=False, showCrosshair=False)
        cv2.destroyWindow("ROI selector")
        return roi


    def tracking_id_from_roi(self, roi, bbox_centers) -> int:
        """Extract tracking id from roi and list of bbox centroids

        Args:
            roi (np.array): coordinate of roi
            bbox_centers (np.array) : list of bbox centrodis

        Returns:
            int: tracking id
        """
        roi_center = np.array([
            int((roi[0]) + (roi[2]/ 2)),
            int((roi[1]) + (roi[3] / 2)),
            ]
        )
        track_id = np.argmin(np.linalg.norm(np.ones((len(bbox_centers),2))*roi_center-bbox_centers, axis=1)) + 1

        return track_id

    def crop_tracked_frame(self, bbox, frame, offset=10, output_width=320):
        """Crop tracked object in frame

        Args:
            bbox: bbox coordinate in tlbr format
            frame: extracting frame 
            offset (int) : Crop offest. Defaults to 10.
            output_width (int): Crop width. Defaults to 320.

        Returns:
            ndarary: Cropped frame
        """
        x1 = int(bbox[0]) - offset
        y1 = int(bbox[1]) - offset
        x2 = int(bbox[2]) + offset
        y2 = int(bbox[3]) + offset
        ratio = frame.shape[0] / frame.shape[1]
        tracking_frame = frame[y1:y2, x1:x2]        
        tracking_frame = cv2.resize(tracking_frame, (int(output_width * ratio), output_width))
        return tracking_frame



