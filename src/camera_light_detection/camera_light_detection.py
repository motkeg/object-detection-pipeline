import cv2
import numpy as np
from collections import deque
from scipy.spatial.distance import mahalanobis
import pandas as pd
import matplotlib.pyplot as plt

STARTING_FRAMES = 10
STD_DEV = 175


def track_light_status(path):
    """
    :param path: path to media
    :return: the function will calculate mahalanobis distance between light intensity of
    frame and the last 'STARTING_FRAMES' before him. if it is grater than 'STD_DEV' it will record a status change
    """
    window_size = STARTING_FRAMES  # Copy of global integer
    light_deque = deque(STARTING_FRAMES * [0], maxlen=STARTING_FRAMES)
    LIGHTS_ON = False
    cap = cv2.VideoCapture(path)
    count = 0
    debug_list = []
    while cap.isOpened():
        success, frame = cap.read()
        count += 1
        if not success:
            cap.release()
            break
        light_from_colorspace = calculate_light_from_colorspace(frame)
        light_algerian = calculate_light_algerian_way(frame)
        light_intensity_list = [light_from_colorspace, light_algerian]
        if count > window_size:
            distance = calc_mahalanobis(light_intensity_list, light_deque)
            debug_list.append(distance)
            if distance >= STD_DEV:
                LIGHTS_ON = not LIGHTS_ON
                print("Light status changed at frame", count, 'from', not LIGHTS_ON, 'to', LIGHTS_ON)
                window_size += count  # Cool-down of 'STARTING_FRAMES'
        light_deque.append(light_intensity_list)

    cap.release()
    return np.array(debug_list)


def calculate_light_from_colorspace(frame):
    """
    :param frame: cv2 frame [H,W,C] in BGR colorspace
    :return: mean value of light channel from HLS colorspace
    """

    imgHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lchannel = imgHLS[:, :, 1]
    lchannel_mean = lchannel.mean()
    return lchannel_mean


def calculate_light_algerian_way(frame):
    """
    :param frame: cv2 frame [H,W,C]
    :return: R+G+B for all pixels divided by 3 then by the total number of pixels
    """
    axis_0, axis_1, _ = frame.shape
    num_of_pixels = axis_0 * axis_1
    light_intensity = frame.sum(axis=0).sum()
    light_intensity /= (3 * num_of_pixels)
    return light_intensity


def calc_mahalanobis(intensities, light_distrib):
    """
    :param intensities: [1 x 2] list contains light intensity values
    :param light_distrib: [N x 2] array of light distances
    :return: mahalanobis distance between two 1-d vectors: intensities and light_dist mean.
             dist^2 = (u-v)IV(u-v).T where IV is the inverse covariance matrix of light_dist
    """
    light_distrib = np.array(light_distrib)
    cov_matrix = np.cov(light_distrib.T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    intensities = np.array(intensities).reshape(-1, 2)
    std_dist = mahalanobis(intensities, light_distrib.mean(axis=0).reshape(-1, 2), inv_cov_matrix)
    return std_dist


if __name__ == '__main__':
    path = '/media/nagler/hdd/Datasets/Sixai Musashi SDV/10.0.0.100_8000_35_3302784CC92E4711A1662993167583F0_/20190829_20190829120651_20190829120830_120651.mp4'
    res = track_light_status(path)
    pd.Series(res).plot()
    plt.show()
