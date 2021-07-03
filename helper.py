import numpy as np
from bounding_box import bounding_box as bb

def drawing_boxes(frame, boxes, class_names, colors=[]):
    dict_color = dict

    if not colors:
        rand_colors = np.random.choice(_colors, len(detecting_objs))
        dict_color = dict(zip(detecting_objs, rand_colors))
    else:
        assert len(colors) == len(detecting_objs), "The number of colors need to match the number of tracking objects: %i color(s) but %i tracking object[s]"%(len(colors), len(detecting_objs))

        dict_color = dict(zip(detecting_objs, colors))

    return_frame = frame.copy()
    for name, box in zip(class_names,boxes):
        (x,y) = (box[0],box[1])
        (w,h) = (box[2],box[3])
        bb.add(return_frame,x,y,x+w,y+h,name,str(dict_color[name]))

    return return_frame