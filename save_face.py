import cv2, json, sys, datetime
import tensorflow as tf
import numpy as np

from face_filter import c_face_filter
from mtcnn_detect import c_MTCNNDetect
from face_attr import c_face_attr_reader

standard_face_size = 160 # 160(weight) * 160(height)
detect_resolution = 80 # 80(weight) * 80(height)

def record_single_face():
    face_imgs =     {"Left" : [], "Right": [], "Center": []}
    face_img_cnt =  {"Left" : 0,  "Right": 0,  "Center": 0}
    face_attrs =    {"Left" : [], "Right": [], "Center": []}

    while True:
        _, frame = vs.read()
        rects, landmarks = face_detect.detect_face(frame, detect_resolution);

        tip = ""; tip_color = 0;
        if(len(rects) <= 0):
            tip = "No face found!"; tip_color = (0, 0, 255);
        else:
            face, direction = the_filter.filter_standard_face(frame, standard_face_size, landmarks[0]);# save 1 face only
            if len(face) == standard_face_size and len(face[0]) == standard_face_size:
                face_imgs[direction].append(face)
                face_img_cnt[direction] += 1
                tip = "Recording..."; tip_color = (0, 255, 0)
            else:
                tip = "Filter face failed!"; tip_color = (0, 255, 255)

        if(face_img_cnt["Left"] > 0 and face_img_cnt["Right"] > 0 and face_img_cnt["Center"] > 0):
            tip = "Press 's' to save this record\nPress 'c' to clear this record"; tip_color = (0, 255, 255)
            
        cv2.putText(frame, tip, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, tip_color, 2, cv2.LINE_AA)
        cv2.imshow("Press 'q' to exit", frame)
        key = cv2.waitKey(1) & 0xFF
        if (key == ord("q")):
            return -1
        elif (key == ord("c")):
            return 0
        elif (key == ord("s")):
            break

    if(face_img_cnt["Left"] == 0 or face_img_cnt["Right"] == 0 or face_img_cnt["Center"] == 0):
        return 0

    file_name = './faces/Face.' + datetime.datetime.now().strftime("%Y-%m-%d.%H-%M-%S") + '.attr'
    print('-------------- ' + file_name + ' --------------')
    for key in face_img_cnt:
        print(key + " image count: " + str(face_img_cnt[key]))

    for key in face_imgs:
        face_attrs[key] = [np.mean(the_face_attrs_reader.get_face_attr(face_imgs[key]),axis=0).tolist()]
    f = open(file_name, 'w')
    f.write(json.dumps(face_attrs))
    return 1

the_face_attrs_reader = c_face_attr_reader(standard_face_size)
the_filter = c_face_filter()
face_detect = c_MTCNNDetect(tf.Graph(), scale_factor=2) #scale_factor, rescales image for faster detection
vs = cv2.VideoCapture(0)

ret = 0
while ret >= 0:
    ret = record_single_face()