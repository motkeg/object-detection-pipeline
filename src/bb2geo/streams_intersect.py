import cv2
import os
import numpy as np
from src.bb2geo.utils import create_colors_list


class StreamIntersectCalculator(object):

    def __init__(self, map_path, *streams_matrix):

        self.matrix = dict(enumerate(streams_matrix))
        self.map = cv2.imread(map_path)



    def get_bbox_from_cnts(self,cnts):
        bboxes = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
        bboxes.append([[x,y],[x+w,y+h]])

        return bboxes

    def transform(self,cnts,matrix):
        tras_cnts = list(map(lambda c :cv2.perspectiveTransform(np.array([c]), matrix[0]),cnts))

        return tras_cnts


    def calculate_intersection(self, *frames_cnts):
        '''

        :param frames_cnts: cv2.findContours outputs for # no of frames
        :return: [cv2.findContours] corresponding to teh map
        '''
        #todo: test on 2 streams
        map_cnts = []
        for i, cnts in enumerate(frames_cnts):
            map_cnts.append(cv2.perspectiveTransform(np.array([cnts]), self.matrix[i]))

        return map_cnts

    def draw_on_map(self, cnts, colors=None, view_map=True, cnts_threshold_interval=None):
        '''

        :param view_map: bool - see the drawing on map or not
        :param colors: Array of (b,g,r)
        :param cnts: Array of map contours
        :return:
        '''

        if colors is None:
            self.colors = create_colors_list(len(cnts))
        else:
            self.colors = colors * len(cnts)

        trasformed_cnts = self.transform(cnts,self.matrix[0])
        for i, c in enumerate(trasformed_cnts):
            if cnts_threshold_interval:
                if cv2.contourArea(c) < cnts_threshold_interval[0] or \
                    cv2.contourArea(c) > cnts_threshold_interval[1]:
                    continue

            area = cv2.contourArea(c)
            (x, y, w, h) = cv2.boundingRect(c)

            cv2.rectangle(self.map, (x, y), (x + w, y + h), self.colors[i], 2)
            print(f'Area of contour {i} : {area}')


        if view_map:
            cv2.imshow('Map', self.map)
            key = cv2.waitKey(200)
            if key & 0xFF == 27:
                cv2.destroyAllWindows()


    def save_map_image(self,path_to_save="./"):
        cv2.imwrite(os.path.join(path_to_save,'map_with_contours.jpg'),self.map)
