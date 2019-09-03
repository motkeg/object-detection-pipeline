
import cv2
import numpy as np
from src.bb2geo.simple_transform import print_frame_loc_on_map


def load_matrix_test(path_to_mat,path_to_map,):
    '''
     we want to drew the following:  [(879, 359), (1017, 360), (1148, 362)]
     on map
    :return:
    '''


    Mat = np.load(path_to_mat)
    print('transform Matrix is:\n' ,Mat)
    map_img = cv2.imread(path_to_map)

    print_frame_loc_on_map([[878, 360], [1016, 362], [1147, 360]],Mat,map_img,color=(0,255,0))
    # print_frame_loc_on_map((1017, 360),map,geo_model,color=(0,0,0))
    # print_frame_loc_on_map((1148, 362),map,geo_model,color=(0,255,0))







