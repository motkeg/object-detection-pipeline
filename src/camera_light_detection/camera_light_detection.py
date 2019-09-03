import cv2
import numpy as np
from collections import deque
from scipy.spatial.distance import mahalanobis
import pandas as pd
import matplotlib.pyplot as plt

STARTING_FRAMES = 10
STD_DEV = 145


def track_light_status(path):
    """
    :param path: path to media
    :return: the function will calculate mahalanobis distance between light intensity of
    frame and the previous 'STARTING_FRAMES'. if it is grater than 'STD_DEV' and the area of the abs diff
    between the last two frames is grater than 10% it will record a status change
    """
    window_size = STARTING_FRAMES  # Copy of global integer
    light_deque = deque(STARTING_FRAMES * [0], maxlen=STARTING_FRAMES)
    cap = cv2.VideoCapture(path)
    count = 0
    debug_list = []  # For debugging purposes only!
    suspected = []  # For debugging purposes only!
    frame = None
    while cap.isOpened():
        last_frame = frame
        success, frame = cap.read()
        count += 1
        if not success:
            cap.release()
            break
        light_from_colorspace = calculate_light_from_colorspace(frame)
        light_algerian = calculate_light_algerian_way(frame)
        light_intensity_list = [light_from_colorspace, light_algerian]
        if count > window_size:

            # 1. record the mark of the difference between the current frame value and the previous frames mean value
            direction = direction_from_mean(np.array(light_intensity_list), np.array(light_deque))
            # 2. record the distance between the current frame and the previous frames distribution
            distance = calc_mahalanobis(light_intensity_list, light_deque)

            # debug_list.append([distance, direction])  # Debug

            # 3. record the area of the absolute difference between the last two frames
            area = light_area_effect([last_frame, frame])

            if distance >= STD_DEV and direction > 0 and area > 0.1:

                #suspected.append([count, last_frame, frame])  # Debug

                print("Light status changed at frame", count)
                window_size += count  # Cool-down of 'STARTING_FRAMES'. needs to be changed with "cap.release()" to
                # break the loop
        light_deque.append(light_intensity_list)

    cap.release()

    # create_diff_mask(suspected)  # Debug
    return np.array(debug_list)  # Debug. replace with anything else needed.


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
    :param light_distrib: [N x 2] list of light distances
    :return: mahalanobis distance between two 1-d vectors: intensities and light_dist mean.
             dist^2 = (u-v)IV(u-v).T where IV is the inverse covariance matrix of light_dist
    """
    light_distrib = np.array(light_distrib)
    cov_matrix = np.cov(light_distrib.T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    intensities = np.array(intensities).reshape(-1, 2)
    std_dist = mahalanobis(intensities, light_distrib.mean(axis=0).reshape(-1, 2), inv_cov_matrix)
    return std_dist


def direction_from_mean(light_intensity, intensities_vector):
    """

    :param light_intensity: the current light intensity measured
    :param intensities_vector: the last frames light intensities
    :return: will return postive number if the current light intensity is bigger than average,
             (lighting -> 1) (occlusion -> -1)
    """
    mean_vector = intensities_vector.mean()
    return 1 if light_intensity.mean() > mean_vector else -1


"""
For debugging purposes only.
"""


def create_diff_mask(frames_list):

    """
    :return: will display the last two frames, and their differences in order to try and see which objects
                         moved and affected lighting.
    """
    from PIL import Image
    for _, val in enumerate(frames_list):
        diff = cv2.absdiff(val[1], val[2])
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        imask = mask > 10
        canvas = np.zeros_like(val[2], np.uint8)
        canvas[imask] = val[2][imask]
        Image.fromarray(canvas).show()
        Image.fromarray(val[1]).show()
        Image.fromarray(val[2]).show()
        pass


def light_area_effect(frames):
    """

    :param frames: gets the current frame and the previous frame
    :return: ratio of pixels who were affected by the light change
    """
    diff = cv2.absdiff(frames[0], frames[1])
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    imask = mask > 10
    axis0, axis1 = imask.shape
    return np.sum(imask)/(axis0 * axis1)








if __name__ == '__main__':
    path = '/media/nagler/hdd/Datasets/Sixai Musashi SDV/10.0.0.100_8000_35_3302784CC92E4711A1662993167583F0_/20190829_20190829120651_20190829120830_120651.mp4'
    res = track_light_status(path)
    colors = np.where(res[:, 1] == 1, 'r', 'b')
    testdf = pd.Series(res[:, 0])
    testdf.plot(color='b')
    testdf.iloc[np.where(res[:, 1] == 1)].plot(color='r')
    plt.show()


