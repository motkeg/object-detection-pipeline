
import cv2
import numpy as np


CIRCLE_RADIUS = 5
CIRCLE_THICKNESS = 3

def print_frame_loc_on_map(locations,matrix,map,color=(0, 255, 0)):
    locations = np.array([locations],dtype='float32')
    results = cv2.perspectiveTransform(locations, matrix)

    for p in results[0]:
        cv2.circle(map, tuple(p), CIRCLE_RADIUS, color, thickness=CIRCLE_THICKNESS)

    while 1:
        cv2.imshow('Map', map.astype('uint8'))
        key = cv2.waitKey(20)
        if key & 0xFF == 27:
            cv2.destroyAllWindows()
            break


def four_point_transform(image, src_pts, dst_pts):
    """

    :param image: numpy array. The frame that we wish to calculate the mapping for.
    :param src_pts: numpy array (shape - (4,2)), dtype ="float32 containing the locations of 4 points in src image
    (in image)
    :param dst_pts: numpy array (shape - (4,2)), dtype ="float32 containing the locations of 4 points in dst image
    corresponding to src_pts.
    :return: numpy array. Shape (image.shape[1], image.shape[0], 2) - The (i, j) location in this array will contain
    the corresponding location [i', j'] in the destination image.
    """

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # creating the heat-map
    # for each location insert it's location vector
    frame_pix_loc = [[i, j] for i in range(image.shape[1]) for j in range(image.shape[0])]
    frame_pix_loc = np.array([frame_pix_loc], dtype="float32")

    # get the transformed locations (as a flatten array)
    frame_heatmap = cv2.perspectiveTransform(frame_pix_loc, M)

    # reshape the transformed flatten image array to the frame's dimensions s.t the i, j location will contain
    # a tuple that contains the corresponding location in the dst image
    frame_heatmap = frame_heatmap.reshape(image.shape[1], image.shape[0], 2)

    return frame_heatmap