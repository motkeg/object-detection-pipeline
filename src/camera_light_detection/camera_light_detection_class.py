import cv2
import numpy as np
from collections import deque
from scipy.spatial.distance import mahalanobis


STARTING_FRAMES = 10
STD_DEV = 145
SYNC_THRESHOLD = 0.5
AREA_THR = 0.1


class CameraLightDetection(object):

    def __init__(self, number_of_streams):
        self.frame_count = np.zeros(number_of_streams)
        self.light_deque = [deque(STARTING_FRAMES * [0], maxlen=STARTING_FRAMES) for _ in range(number_of_streams)]
        self.frames = [None for _ in range(number_of_streams)]
        self.last_frames = [None for _ in range(number_of_streams)]
        self.flags = [None for _ in range(number_of_streams)]

    @staticmethod
    def calculate_light_from_colorspace(frame):
        """
        :param frame: cv2 frame [H,W,C] in BGR colorspace
        :return: mean value of light channel from HLS colorspace
        """

        imgHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lchannel = imgHLS[:, :, 1]
        lchannel_mean = lchannel.mean()
        return lchannel_mean

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def direction_from_mean(light_intensity, intensities_vector):
        """

        :param light_intensity: the current light intensity measured
        :param intensities_vector: the last frames light intensities
        :return: will return positive number if the current light intensity is bigger than average,
                 (lighting -> 1) (occlusion -> -1)
        """
        mean_vector = np.array(intensities_vector).mean()
        return 1 if np.array(light_intensity).mean() > mean_vector else -1

    """
    For debugging purposes only.
    """
    @staticmethod
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

    @staticmethod
    def light_area_effect(frames):
        """

        :param frames: gets the current frame and the previous frame
        :return: ratio of pixels who were affected by the light change
        """
        diff = cv2.absdiff(frames[0], frames[1])
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        imask = mask > 10
        axis0, axis1 = imask.shape
        return np.sum(imask) / (axis0 * axis1)

    def is_synced(self):
        """
        :return: 0: Didn't detect light switch in atleast one camera
                -1: Out of sync.
                 1: Synced.
        """
        if None in self.flags:
            return 0
        max_timestamp = np.max(self.flags)
        min_timestamp = np.min(self.flags)
        if max_timestamp-min_timestamp > SYNC_THRESHOLD:
            return -1
        return 1

    def track_light_change(self, img, stream_id, timestamp):

        if self.flags[stream_id]:
            return self.is_synced()

        self.frame_count[stream_id] += 1
        light_intensity_list = [self.calculate_light_from_colorspace(frame=img),
                                self.calculate_light_algerian_way(frame=img)]
        self.last_frames[stream_id] = self.frames[stream_id]
        self.frames[stream_id] = img
        if self.frame_count[stream_id] > STARTING_FRAMES:
            # 1. record the mark of the difference between the current frame value and the previous frames mean value
            direction = self.direction_from_mean(light_intensity_list,
                                                 self.light_deque[stream_id])
            # 2. record the distance between the current frame and the previous frames distribution
            distance = self.calc_mahalanobis(light_intensity_list, self.light_deque[stream_id])

            # 3. record the area of the absolute difference between the last two frames
            area = self.light_area_effect([self.last_frames[stream_id], img])

            if distance >= STD_DEV and direction > 0 and area > AREA_THR:
                self.flags[stream_id] = timestamp

        self.light_deque[stream_id].append(light_intensity_list)
        return self.is_synced()


if __name__ == '__main__':
    test = CameraLightDetection(2)
    print("Debug")
    pass




