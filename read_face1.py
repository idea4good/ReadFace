import cv2, json, sys, datetime
import tensorflow as tf
import numpy as np

from face_filter import c_face_filter
from mtcnn_quick import MTCNN # customize mtcnn for quickly performance
from face_attr import c_face_attr_reader

standard_faces_size = 160 # 160(weight) * 160(height)
detect_resolution = 40 # 80(weight) * 80(height)

def search_face(face_attr, face_directions, diff_thres = 0.6, odds_thres = 70):
    ret = [];
    for (i, attr) in enumerate(face_attr):
        name = "??";
        diff = sys.maxsize
        for person in the_database.keys():
            person_data = the_database[person][face_directions[i]];
            for data in person_data:
                cur_diff = np.sqrt(np.sum(np.square(data - attr)))
                if(cur_diff < diff):
                    diff = cur_diff
                    name = person
        odds =  round(min(100, 100 * diff_thres / diff), 1)

        if odds <= odds_thres :
            ret.append("??")
            print("??@ " + datetime.datetime.now().strftime("%H-%M-%S"))
        else:
            ret.append(name + ":" + str(odds) + "%")
            print(name + '@ ' + datetime.datetime.now().strftime("%H-%M-%S"))
    return ret

the_face_attr_reader = c_face_attr_reader(standard_faces_size)
the_filter = c_face_filter();
face_detect = MTCNN(tf.Graph(), scale_factor=2); #scale_factor, rescales image for faster detection
the_database = json.loads(open('./face_database.txt','r').read());

vs = cv2.VideoCapture(1);# 0: default; 1: Microsoft lifecam

vs.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

print(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
print(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    _, frame = vs.read();
    rects, keypoints = face_detect.detect_faces(frame, detect_resolution)

    standard_faces = []; face_directions = []
    for (i, rect) in enumerate(rects):
        face, direction = the_filter.filter_standard_face(frame, standard_faces_size, keypoints[i])
        if len(face) == standard_faces_size and len(face[0]) == standard_faces_size:
            standard_faces.append(face)
            face_directions.append(direction)
        else: 
            print("Filter face failed")

    if(len(standard_faces) > 0):
        face_attrs = the_face_attr_reader.get_face_attr(standard_faces)
        person_info = search_face(face_attrs, face_directions);
        for (i,rect) in enumerate(rects):
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255,104,38)) #draw bounding box for the face
            cv2.putText(frame, person_info[i], (rect[0] + rect[2], rect[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,104,38),1,cv2.LINE_AA)

    cv2.imshow("Press 'q' to exit", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break